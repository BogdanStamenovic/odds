from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from email.message import Message
from typing import Any

import pytest

from odds import models, sources
from odds.sources import Hit

OPENALEX = {
    "results": [
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1000/ABC.123",
            "display_name": "Casual Sex Attitudes in East Asia",
            "publication_year": 2019,
            "cited_by_count": 240,
            "language": "en",
            "abstract_inverted_index": {"We": [0], "survey": [1], "attitudes.": [2]},
            "primary_location": {"landing_page_url": "https://journal.example/abc"},
        },
        {"id": "https://openalex.org/W2", "display_name": "", "doi": None},
    ]
}

S2 = {
    "total": 2,
    "data": [
        {
            "paperId": "abc",
            "title": "Casual sex attitudes in East Asia",  # same paper, different case
            "url": "https://www.semanticscholar.org/paper/abc",
            "abstract": "Duplicate of the OpenAlex one.",
            "year": 2019,
            "citationCount": 260,
        },
        {
            "paperId": "def",
            "title": "Hookup Culture Among Korean Students",
            "url": "https://www.semanticscholar.org/paper/def",
            "abstract": None,
            "year": 2021,
            "citationCount": 3,
        },
    ],
}

ARXIV = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>arXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2101.00001v2</id>
    <published>2021-01-01T00:00:00Z</published>
    <title>Dating App Matching
      Dynamics</title>
    <summary>  We model swipes.  </summary>
  </entry>
</feed>"""

WIKI_KO = {
    "batchcomplete": True,
    "query": {
        "pages": [
            {"pageid": 2, "title": "한국의 성문화", "index": 2,
             "extract": "두 번째.",
             "fullurl": "https://ko.wikipedia.org/wiki/B"},
            {"pageid": 1, "title": "연애", "index": 1, "extract": "첫 번째.",
             "fullurl": "https://ko.wikipedia.org/wiki/A"},
        ]
    },
}

REDDIT_TOKEN = {"access_token": "tok123", "token_type": "bearer", "expires_in": 86400}
REDDIT_SEARCH = {
    "kind": "Listing",
    "data": {
        "children": [
            {"kind": "t3", "data": {
                "title": "Dating in Seoul as a foreigner",
                "permalink": "/r/korea/comments/xyz/dating_in_seoul/",
                "subreddit_name_prefixed": "r/korea",
                "selftext": "Long story...",
                "created_utc": 1700000000.0,
            }},
        ]
    },
}


def http_error(url: str, code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, "err", headers, io.BytesIO(b""))


Route = Callable[[urllib.request.Request], Any]


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Route urlopen by host. A value is a payload (dict/bytes), an Exception, or a callable."""
    routes: dict[str, Any] = {}
    calls: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 0) -> io.BytesIO:
        calls.append(req)
        host = urllib.parse.urlsplit(req.full_url).netloc
        key = next((k for k in routes if k == host or req.full_url.startswith(k)), None)
        if key is None:
            raise urllib.error.URLError(f"no route to {host}")
        value = routes[key]
        if callable(value) and not isinstance(value, type):
            value = value(req)
        if isinstance(value, BaseException):
            raise value
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        return io.BytesIO(body)

    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    monkeypatch.setattr(sources, "_reddit_token", None)
    return {"routes": routes, "calls": calls}


def collect() -> tuple[list[str], Callable[[str], None]]:
    msgs: list[str] = []
    return msgs, msgs.append


# ---- adapters ---------------------------------------------------------------


def test_openalex_parses_and_uninverts_abstract(net: dict[str, Any]) -> None:
    net["routes"]["api.openalex.org"] = OPENALEX
    hits = sources.search("q", adapters=["openalex"], warn=lambda m: None)
    assert len(hits) == 1
    h = hits[0]
    assert h.url == "https://doi.org/10.1000/ABC.123"
    assert h.snippet == "We survey attitudes."
    assert (h.year, h.citations, h.kind, h.adapter) == (2019, 240, "study", "openalex")
    assert net["calls"][0].get_header("User-agent") == sources.USER_AGENT


def test_semanticscholar_and_arxiv(net: dict[str, Any]) -> None:
    net["routes"]["api.semanticscholar.org"] = S2
    net["routes"]["export.arxiv.org"] = ARXIV
    hits = sources.search("q", adapters=["semanticscholar", "arxiv"], warn=lambda m: None)
    assert [h.adapter for h in hits] == ["semanticscholar", "semanticscholar", "arxiv"]
    ax = hits[2]
    assert ax.title == "Dating App Matching Dynamics"
    assert ax.snippet == "We model swipes."
    assert ax.year == 2021
    assert hits[1].snippet == ""


def test_arxiv_query_is_anded_terms(net: dict[str, Any]) -> None:
    net["routes"]["export.arxiv.org"] = ARXIV
    sources.search("casual sex, Korea", adapters=["arxiv"], warn=lambda m: None)
    qs = urllib.parse.parse_qs(urllib.parse.urlsplit(net["calls"][0].full_url).query)
    assert qs["search_query"] == ["all:casual AND all:sex AND all:Korea"]


def test_wikipedia_any_language_sorted_by_index(net: dict[str, Any]) -> None:
    net["routes"]["ko.wikipedia.org"] = WIKI_KO
    hits = sources.search("연애", adapters=["wikipedia"], lang="ko", warn=lambda m: None)
    assert [h.url for h in hits] == ["https://ko.wikipedia.org/wiki/A",
                                     "https://ko.wikipedia.org/wiki/B"]
    assert all(h.lang == "ko" and h.kind == "wiki" for h in hits)


def test_wikipedia_rejects_hostile_lang(net: dict[str, Any]) -> None:
    msgs, warn = collect()
    assert sources.search("x", adapters=["wikipedia"], lang="evil.com/", warn=warn) == []
    assert not net["calls"]
    assert "language code" in msgs[0]


# ---- reddit -----------------------------------------------------------------


def test_reddit_unavailable_without_creds(net: dict[str, Any],
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ODDS_REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("ODDS_REDDIT_CLIENT_SECRET", raising=False)
    status = sources.available()
    assert status["reddit"] == "register a Reddit app and set ODDS_REDDIT_CLIENT_ID/SECRET"
    assert status["openalex"] == "ok"
    msgs, warn = collect()
    assert sources.search("x", adapters=["reddit"], warn=warn) == []
    assert not net["calls"]
    assert "reddit unavailable" in msgs[0]


def test_reddit_oauth_flow(net: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODDS_REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("ODDS_REDDIT_CLIENT_SECRET", "secret")
    net["routes"]["https://www.reddit.com/api/v1/access_token"] = REDDIT_TOKEN
    net["routes"]["oauth.reddit.com"] = REDDIT_SEARCH
    hits = sources.search("seoul dating", adapters=["reddit"], warn=lambda m: None)
    assert len(hits) == 1
    assert hits[0].url == "https://www.reddit.com/r/korea/comments/xyz/dating_in_seoul/"
    assert hits[0].kind == "reddit" and hits[0].year == 2023
    token_req, search_req = net["calls"]
    assert token_req.data == b"grant_type=client_credentials"
    assert token_req.get_header("Authorization", "").startswith("Basic ")
    assert search_req.get_header("Authorization") == "bearer tok123"
    # Only the official OAuth host is ever searched.
    assert all(urllib.parse.urlsplit(r.full_url).netloc != "www.reddit.com"
               or r.full_url.endswith("/api/v1/access_token") for r in net["calls"])


# ---- fan-out, dedupe, failure ----------------------------------------------


def test_dedupe_across_adapters_by_url_and_study_title(net: dict[str, Any]) -> None:
    net["routes"]["api.openalex.org"] = OPENALEX
    net["routes"]["api.semanticscholar.org"] = S2
    hits = sources.search("q", adapters=["openalex", "semanticscholar"], warn=lambda m: None)
    assert [h.title for h in hits] == ["Casual Sex Attitudes in East Asia",
                                       "Hookup Culture Among Korean Students"]
    assert hits[0].citations == 260  # the duplicate's better count is kept


def test_normalize_url() -> None:
    n = sources.normalize_url
    assert n("http://www.Example.com/a/?utm_source=x&b=1#frag") == "https://example.com/a?b=1"
    assert n("https://dx.doi.org/10.1/ABC") == n("https://doi.org/10.1/abc")
    assert n("http://arxiv.org/abs/2101.00001v2") == n("https://arxiv.org/abs/2101.00001v1")


@pytest.mark.parametrize("error", [
    http_error("https://api.openalex.org/works", 500),
    urllib.error.URLError("Name or service not known"),
    TimeoutError("timed out"),
])
def test_dead_adapter_warns_and_others_survive(net: dict[str, Any], error: Exception) -> None:
    net["routes"]["api.openalex.org"] = error
    net["routes"]["ko.wikipedia.org"] = WIKI_KO
    msgs, warn = collect()
    hits = sources.search("q", adapters=["openalex", "wikipedia"], lang="ko", warn=warn)
    assert [h.adapter for h in hits] == ["wikipedia", "wikipedia"]
    assert len(msgs) == 1 and "openalex" in msgs[0]


def test_403_is_a_stop_not_retried(net: dict[str, Any]) -> None:
    net["routes"]["api.openalex.org"] = http_error("https://api.openalex.org/works", 403)
    msgs, warn = collect()
    assert sources.search("q", adapters=["openalex"], warn=warn) == []
    assert len(net["calls"]) == 1
    assert "403" in msgs[0]


def test_429_retries_once_honoring_capped_retry_after(net: dict[str, Any],
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(sources.time, "sleep", slept.append)
    attempts = iter([http_error("u", 429, retry_after="120"), S2])
    net["routes"]["api.semanticscholar.org"] = lambda req: next(attempts)
    hits = sources.search("q", adapters=["semanticscholar"], warn=lambda m: None)
    assert len(hits) == 2
    assert slept == [sources.RETRY_AFTER_CAP]


def test_429_twice_gives_up(net: dict[str, Any]) -> None:
    net["routes"]["api.semanticscholar.org"] = lambda req: http_error("u", 429)
    msgs, warn = collect()
    assert sources.search("q", adapters=["semanticscholar"], warn=warn) == []
    assert len(net["calls"]) == 2
    assert "429" in msgs[0]


def test_bad_payload_and_unknown_adapter_warn(net: dict[str, Any]) -> None:
    net["routes"]["api.openalex.org"] = b"<html>not json</html>"
    net["routes"]["export.arxiv.org"] = b"<feed><unclosed>"
    msgs, warn = collect()
    hits = sources.search("q", adapters=["openalex", "arxiv", "nope"], warn=warn)
    assert hits == []
    assert len(msgs) == 3
    assert any("unknown adapter 'nope'" in m for m in msgs)


def test_limit_is_respected(net: dict[str, Any]) -> None:
    net["routes"]["api.semanticscholar.org"] = S2
    assert len(sources.search("q", adapters=["semanticscholar"], limit=1,
                              warn=lambda m: None)) == 1


# ---- quality / Source --------------------------------------------------------


def test_quality_ordering() -> None:
    q = sources.default_quality
    cited = Hit("t", "u", "study", citations=2000)
    uncited = Hit("t", "u", "study", citations=0)
    assert 0.85 <= q(cited) <= 0.9
    assert q(cited) > q("statistic") > q(uncited) >= q("wiki") > q("news") > q("guide") \
        > q("other") > q("reddit") == q("forum")
    assert q(Hit("t", "u", "study", citations=100)) > q(Hit("t", "u", "study", citations=10))
    assert 0.0 <= q("nonsense-kind") <= 1.0


def test_to_source() -> None:
    hit = Hit("Title", "https://x.org/p", "study", snippet="abs", year=2020, citations=10,
              lang="en", adapter="openalex")
    src = sources.to_source(hit, "S3")
    assert isinstance(src, models.Source)
    assert (src.id, src.url, src.title, src.kind) == ("S3", "https://x.org/p", "Title", "study")
    assert src.quality == sources.default_quality(hit)
    assert src.note == "via openalex, 2020, 10 citations: abs"
    assert sources.to_source(Hit("t", "u", "bogus"), "S1").kind == "other"
