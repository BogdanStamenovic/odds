"""Keyless source adapters: scholarly indexes, Wikipedia in any language, and Reddit.

These give the pipeline real, citable sources even when the research engine has
no web search of its own (a local Ollama model), and give claude/codex
researchers scholarly leads to start from. Standard library only, on purpose:
this module sits on the network boundary and should never be the reason an
install fails.

Every adapter identifies itself honestly (see `USER_AGENT`). A 403 is treated
as a real stop -- the adapter warns and returns nothing. There is no header
spoofing, no user-agent rotation and no proxying, and there will not be.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from odds import models

USER_AGENT = "odds/0.1 (+https://github.com/BogdanStamenovic/odds)"

# One retry on 429, and never sleep longer than this no matter what Retry-After
# says: a research run fans out many searches and a single adapter must not
# stall it for a minute.
RETRY_AFTER_CAP = 5.0

Warn = Callable[[str], None]


@dataclass
class Hit:
    title: str
    url: str
    kind: str  # one of models.SOURCE_KINDS
    snippet: str = ""
    year: int | None = None
    citations: int | None = None
    lang: str = "en"
    adapter: str = ""


class AdapterError(Exception):
    """A failure the fan-out reports as a warning instead of raising."""


class Blocked(AdapterError):
    """The server refused us (403). A real stop, not something to route around."""


# ---- HTTP -----------------------------------------------------------------


def _request(
    url: str,
    *,
    timeout: float,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> bytes:
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json, application/xml"}
    req_headers.update(headers or {})
    host = urllib.parse.urlsplit(url).netloc
    for attempt in (0, 1):
        req = urllib.request.Request(url, data=data, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body: bytes = resp.read()
                return body
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                time.sleep(_retry_after(exc.headers.get("Retry-After") if exc.headers else None))
                continue
            if exc.code == 429:
                raise AdapterError(f"{host} rate-limited us (HTTP 429) twice; skipping") from exc
            if exc.code == 403:
                raise Blocked(f"{host} refused the request (HTTP 403); not retrying") from exc
            raise AdapterError(f"{host} answered HTTP {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise AdapterError(f"{host} unreachable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise AdapterError(f"{host} timed out after {timeout:g}s") from exc
    raise AssertionError("unreachable")


def _retry_after(value: str | None) -> float:
    try:
        seconds = float(value) if value is not None else 1.0
    except ValueError:
        # HTTP-date form; not worth parsing for a capped wait.
        seconds = RETRY_AFTER_CAP
    return max(0.0, min(seconds, RETRY_AFTER_CAP))


def _get_json(url: str, *, timeout: float, headers: dict[str, str] | None = None) -> Any:
    body = _request(url, timeout=timeout, headers=headers)
    try:
        return json.loads(body)
    except ValueError as exc:
        raise AdapterError(f"{urllib.parse.urlsplit(url).netloc} sent invalid JSON") from exc


def _clean(text: str | None, limit: int = 500) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ---- adapters ---------------------------------------------------------------


def _openalex(query: str, lang: str, limit: int, timeout: float) -> list[Hit]:
    params = urllib.parse.urlencode({
        "search": query,
        "per-page": str(limit),
        "select": "id,doi,display_name,publication_year,cited_by_count,language,"
                  "abstract_inverted_index,primary_location",
    })
    data = _get_json(f"https://api.openalex.org/works?{params}", timeout=timeout)
    hits = []
    for work in data.get("results") or []:
        title = _clean(work.get("display_name"))
        landing = ((work.get("primary_location") or {}).get("landing_page_url"))
        url = work.get("doi") or landing or work.get("id") or ""
        if not title or not url:
            continue
        hits.append(Hit(
            title=title, url=url, kind="study",
            snippet=_clean(_uninvert(work.get("abstract_inverted_index"))),
            year=work.get("publication_year"), citations=work.get("cited_by_count"),
            lang=work.get("language") or "en", adapter="openalex",
        ))
    return hits


def _uninvert(index: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]} for licensing reasons."""
    if not index:
        return ""
    words: dict[int, str] = {}
    for word, positions in index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def _semanticscholar(query: str, lang: str, limit: int, timeout: float) -> list[Hit]:
    # Unauthenticated callers share one global pool (on the order of 1 req/s
    # across everyone), so a 429 here is routine, not a fault. `_request` waits
    # once, then the fan-out reports it and moves on.
    params = urllib.parse.urlencode({
        "query": query, "limit": str(limit), "fields": "title,url,abstract,year,citationCount",
    })
    data = _get_json(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}",
                     timeout=timeout)
    hits = []
    for paper in data.get("data") or []:
        title = _clean(paper.get("title"))
        url = paper.get("url") or (
            f"https://www.semanticscholar.org/paper/{paper['paperId']}"
            if paper.get("paperId") else "")
        if not title or not url:
            continue
        hits.append(Hit(
            title=title, url=url, kind="study", snippet=_clean(paper.get("abstract")),
            year=paper.get("year"), citations=paper.get("citationCount"),
            adapter="semanticscholar",
        ))
    return hits


_ATOM = "{http://www.w3.org/2005/Atom}"


def _arxiv(query: str, lang: str, limit: int, timeout: float) -> list[Hit]:
    terms = re.findall(r"\w+", query)
    if not terms:
        return []
    params = urllib.parse.urlencode({
        "search_query": " AND ".join(f"all:{t}" for t in terms),
        "start": "0", "max_results": str(limit),
    })
    body = _request(f"https://export.arxiv.org/api/query?{params}", timeout=timeout)
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise AdapterError("export.arxiv.org sent invalid XML") from exc
    hits = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = _clean(entry.findtext(f"{_ATOM}title"))
        url = (entry.findtext(f"{_ATOM}id") or "").strip()
        if not title or not url:
            continue
        published = entry.findtext(f"{_ATOM}published") or ""
        hits.append(Hit(
            title=title, url=url, kind="study",
            snippet=_clean(entry.findtext(f"{_ATOM}summary")),
            year=int(published[:4]) if published[:4].isdigit() else None,
            adapter="arxiv",
        ))
    return hits


# Interpolated into a hostname, so it must be a plain language code.
_LANG_RE = re.compile(r"^[a-z]{2,3}(-[a-z0-9]{2,8})*$")


def _wikipedia(query: str, lang: str, limit: int, timeout: float) -> list[Hit]:
    lang = lang.lower()
    if not _LANG_RE.match(lang):
        raise AdapterError(f"not a Wikipedia language code: {lang!r}")
    # generator=search + prop=extracts does list=search and the intro extracts
    # in one request. The extracts module caps intros at 20 pages per call.
    params = urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": "2",
        "generator": "search", "gsrsearch": query, "gsrlimit": str(min(limit, 20)),
        "prop": "extracts|info", "exintro": "1", "explaintext": "1", "exsentences": "3",
        "exlimit": "max", "inprop": "url",
    })
    data = _get_json(f"https://{lang}.wikipedia.org/w/api.php?{params}", timeout=timeout)
    pages = (data.get("query") or {}).get("pages") or []
    if isinstance(pages, dict):  # formatversion=1 shape, in case a mirror ignores v2
        pages = list(pages.values())
    pages.sort(key=lambda p: p.get("index", 0))  # generator results arrive unordered
    hits = []
    for page in pages:
        title = page.get("title") or ""
        url = page.get("fullurl") or (
            f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}")
        if not title:
            continue
        hits.append(Hit(title=title, url=url, kind="wiki", snippet=_clean(page.get("extract")),
                        lang=lang, adapter="wikipedia"))
    return hits


_REDDIT_ID = "ODDS_REDDIT_CLIENT_ID"
_REDDIT_SECRET = "ODDS_REDDIT_CLIENT_SECRET"
_reddit_token: tuple[str, float] | None = None
_reddit_lock = threading.Lock()


def _reddit_available() -> str:
    if os.environ.get(_REDDIT_ID) and os.environ.get(_REDDIT_SECRET):
        return "ok"
    return f"register a Reddit app and set {_REDDIT_ID}/SECRET"


def _reddit_bearer(timeout: float) -> str:
    """App-only OAuth (client_credentials). Cached until shortly before expiry."""
    global _reddit_token
    with _reddit_lock:
        if _reddit_token and _reddit_token[1] > time.time():
            return _reddit_token[0]
        creds = f"{os.environ[_REDDIT_ID]}:{os.environ[_REDDIT_SECRET]}".encode()
        body = _request(
            "https://www.reddit.com/api/v1/access_token", timeout=timeout,
            data=b"grant_type=client_credentials",
            headers={"Authorization": "Basic " + base64.b64encode(creds).decode()},
        )
        try:
            payload = json.loads(body)
            token = payload["access_token"]
        except (ValueError, KeyError) as exc:
            raise AdapterError("reddit token endpoint gave no access_token") from exc
        _reddit_token = (token, time.time() + float(payload.get("expires_in", 3600)) - 60)
        return str(token)


def _reddit(query: str, lang: str, limit: int, timeout: float) -> list[Hit]:
    reason = _reddit_available()
    if reason != "ok":
        raise AdapterError(reason)
    token = _reddit_bearer(timeout)
    params = urllib.parse.urlencode({
        "q": query, "limit": str(limit), "sort": "relevance", "type": "link", "raw_json": "1",
    })
    data = _get_json(f"https://oauth.reddit.com/search?{params}", timeout=timeout,
                     headers={"Authorization": f"bearer {token}"})
    hits = []
    for child in (data.get("data") or {}).get("children") or []:
        post = child.get("data") or {}
        title = _clean(post.get("title"))
        permalink = post.get("permalink") or ""
        if not title or not permalink:
            continue
        created = post.get("created_utc")
        sub = post.get("subreddit_name_prefixed") or ""
        hits.append(Hit(
            title=f"{title} ({sub})" if sub else title,
            url=f"https://www.reddit.com{permalink}", kind="reddit",
            snippet=_clean(post.get("selftext")),
            year=time.gmtime(created).tm_year if isinstance(created, (int, float)) else None,
            adapter="reddit",
        ))
    return hits


@dataclass(frozen=True)
class Adapter:
    name: str
    fetch: Callable[[str, str, int, float], list[Hit]]  # (query, lang, limit, timeout)
    check: Callable[[], str] = lambda: "ok"


ADAPTERS: dict[str, Adapter] = {
    a.name: a for a in (
        Adapter("openalex", _openalex),
        Adapter("semanticscholar", _semanticscholar),
        Adapter("arxiv", _arxiv),
        Adapter("wikipedia", _wikipedia),
        Adapter("reddit", _reddit, _reddit_available),
    )
}


def available() -> dict[str, str]:
    """name -> "ok", or why the adapter cannot run. Checks config only, no network."""
    return {name: adapter.check() for name, adapter in ADAPTERS.items()}


# ---- fan-out ----------------------------------------------------------------


def search(
    query: str,
    *,
    adapters: Sequence[str] | None = None,
    lang: str = "en",
    limit: int = 5,
    warn: Warn | None = None,
    timeout: float = 15.0,
) -> list[Hit]:
    """Query the adapters in parallel. Never raises on a network failure; warns instead.

    Results keep adapter order (as given, else `ADAPTERS` order) and each
    adapter's own ranking, deduplicated by normalized URL and, for studies, by
    title -- the same paper turns up in OpenAlex, Semantic Scholar and arXiv.
    """
    say = warn or (lambda msg: print(f"odds: {msg}", file=sys.stderr))
    names = list(adapters) if adapters is not None else list(ADAPTERS)
    runnable = []
    for name in names:
        adapter = ADAPTERS.get(name)
        if adapter is None:
            say(f"sources: unknown adapter {name!r}; known: {', '.join(ADAPTERS)}")
            continue
        status = adapter.check()
        if status != "ok":
            say(f"sources: {name} unavailable: {status}")
            continue
        runnable.append(adapter)
    if not runnable or not query.strip():
        return []

    def run(adapter: Adapter) -> list[Hit]:
        try:
            return adapter.fetch(query, lang, limit, timeout)[:limit]
        except AdapterError as exc:
            say(f"sources: {adapter.name}: {exc}")
        except Exception as exc:  # noqa: BLE001 -- a broken adapter must not kill the search
            say(f"sources: {adapter.name} failed: {type(exc).__name__}: {exc}")
        return []

    with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
        results = list(pool.map(run, runnable))
    return _dedupe([hit for batch in results for hit in batch])


_TRACKING = re.compile(r"^(utm_|fbclid$|gclid$|ref$|ref_src$)")


def normalize_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    if host == "dx.doi.org":
        host = "doi.org"
    path = parts.path.rstrip("/") or "/"
    if host == "doi.org":
        path = path.lower()  # DOIs are case-insensitive
    if host.endswith("arxiv.org"):
        host = "arxiv.org"
        path = re.sub(r"v\d+$", "", path)  # versions of one preprint are one source
    query = urllib.parse.urlencode(sorted(
        (k, v) for k, v in urllib.parse.parse_qsl(parts.query) if not _TRACKING.match(k)))
    return urllib.parse.urlunsplit(("https", host, path, query, ""))


def _title_key(title: str) -> str:
    return re.sub(r"[\W_]+", "", title.lower())


def _dedupe(hits: list[Hit]) -> list[Hit]:
    by_url: dict[str, Hit] = {}
    by_title: dict[str, Hit] = {}
    out = []
    for hit in hits:
        url_key = normalize_url(hit.url)
        title_key = _title_key(hit.title) if hit.kind == "study" else ""
        kept = by_url.get(url_key) or (by_title.get(title_key) if title_key else None)
        if kept is not None:
            # Keep the first, but let a later duplicate fill in what it lacked.
            kept.snippet = kept.snippet or hit.snippet
            kept.year = kept.year if kept.year is not None else hit.year
            if hit.citations is not None:
                kept.citations = max(kept.citations or 0, hit.citations)
            continue
        by_url[url_key] = hit
        if title_key:
            by_title[title_key] = hit
        out.append(hit)
    return out


# ---- quality ----------------------------------------------------------------

_KIND_QUALITY = {
    "study": 0.7, "statistic": 0.8, "wiki": 0.6, "news": 0.55,
    "guide": 0.45, "forum": 0.3, "reddit": 0.3, "other": 0.4,
}


def default_quality(hit_or_kind: Hit | str) -> float:
    """Prior on how much a claim from this source should move belief, 0..1.

    Studies with a known citation count are scaled from 0.6 (uncited) to 0.9
    (~1000+ citations), log-scaled. A citation count is a popularity screen,
    not a verdict on the paper, which is why it only nudges within a band.
    """
    if isinstance(hit_or_kind, str):
        return _KIND_QUALITY.get(hit_or_kind, _KIND_QUALITY["other"])
    hit = hit_or_kind
    if hit.kind == "study" and hit.citations is not None:
        return round(0.6 + 0.3 * min(1.0, math.log10(1 + max(0, hit.citations)) / 3), 3)
    return _KIND_QUALITY.get(hit.kind, _KIND_QUALITY["other"])


def to_source(hit: Hit, source_id: str) -> models.Source:
    meta = [f"via {hit.adapter}" if hit.adapter else ""]
    if hit.year is not None:
        meta.append(str(hit.year))
    if hit.citations is not None:
        meta.append(f"{hit.citations} citations")
    note = ", ".join(m for m in meta if m)
    if hit.snippet:
        note = f"{note}: {hit.snippet}" if note else hit.snippet
    kind = hit.kind if hit.kind in models.SOURCE_KINDS else "other"
    return models.Source(id=source_id, url=hit.url, title=hit.title, kind=kind,
                         quality=default_quality(hit), lang=hit.lang, note=note)
