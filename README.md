# odds

`odds` researches a silly question the way an analyst would research a market.
Ask it "getting a one night stand in Korea" or "how do I get upgraded to
business class". It sends LLM researchers to the web and to scholarly indexes,
sets competing hypotheses against each other, has a critic try to break the
result, and turns stage-by-stage probability ranges into odds. What you get
back is a **play**: ranked strategies, each with signs it is working, signs to
bail, and cheap tests that tell you mid-play which hypothesis is winning.

The reference is Patrick Jane from *The Mentalist*, not a search engine. Jane
isn't psychic and doesn't know more facts than anyone else. He reads what the
people on the other side want and fear, notices the detail everyone waves
away, and sets up the situation instead of just predicting it.

## How it works

```
reframe ─▶ plan ─▶ research (parallel) ─▶ synthesize ─▶ critic ─┐
   │         ▲                                   ▲               │ revise
   │         └──── new subquestions ◀────────────┼───────────────┘
   ▼                                             │
 asks you only if the                 asks you about gaps that would
 whole market depends on it           change the answer, then re-synthesizes
                                                 │
                                          Monte Carlo + levers ─▶ report
```

| Step | What it does |
|---|---|
| 1. Reframe | Defines success precisely, the time window, the reference class, the local languages, who decides the outcome. It asks you an unknown only if the unknown changes the whole market (e.g. gender for a dating question); everything else becomes a stated assumption. |
| 2. Plan | Splits the question into subquestions (base rate, the other side's psychology, market map, local norms, logistics, legal/safety risk, a myth to test), each with short English and local-language queries, plus 3–5 **competing hypotheses**. |
| 3. Research | One researcher per subquestion, in parallel, spread round-robin across the engines you name. Each starts from leads pulled out of OpenAlex / Semantic Scholar / arXiv / Wikipedia, then searches the web itself. Every finding is tagged `sourced`, `inferred` or `speculation`, and records which hypotheses it supports or contradicts. |
| 4–6. Synthesize | Base rate, "the other side" (what the counterparts want, fear and signal), hypothesis posteriors judged by *diagnostic* evidence, and 2–5 strategies. Each strategy is a chain of conditional stages with probability ranges tied to evidence ids. |
| 5. Critic | A separate pass whose only job is to break the analysis: anecdotes passed off as rates, survivorship bias, sources with a motive, ranges that are too narrow. It downgrades evidence and can send new subquestions back to research. |
| 7. Estimate | Each stage range is read as an 80% interval on a logit-normal distribution and simulated 20,000 times (seeded), giving p10/p50/p90 per attempt and over your window. A tornado swing per stage ranks the levers. |
| 8. Ask | Gaps whose answer would change the result are put to you (in a terminal), and the run re-synthesizes with your answers. |
| 9. Report | A self-contained HTML report plus a terminal summary. |

Every step writes `run.json` before the next one starts, so a crash keeps
what was already found, and `odds followup` can build on a finished run.

### Engines

| Engine | Web | How it is called |
|---|---|---|
| `claude[:model]` | yes | `claude -p` restricted to WebSearch + WebFetch, with no MCP servers and no settings loaded |
| `codex[:model]` | yes | `codex --search exec -s read-only` |
| `ollama[:model]` | no | local HTTP API, JSON mode; it only sees the source adapters' leads |

All of them are keyless: the two CLIs use your existing logins and Ollama runs
locally. Defaults: `claude:opus` for the judgement calls (reframe, plan,
synthesis, critic), `claude:sonnet` researchers.

### Sources

| Adapter | Needs | Notes |
|---|---|---|
| OpenAlex | nothing | broad, but relevance is loose: in the live test 2 of 5 papers were on topic |
| Semantic Scholar | nothing | rate-limits hard without a key and drops out of some runs |
| arXiv | nothing | matches every query word, so it needs short keyword queries |
| Wikipedia | nothing | any language; this is how local-language context gets in without a web engine |
| Reddit | a free Reddit app | official OAuth API only; set `ODDS_REDDIT_CLIENT_ID` / `ODDS_REDDIT_CLIENT_SECRET` |

A 403, paywall, login wall or anti-bot page is treated as a real stop: it gets
logged as a dead end. Nothing spoofs headers or routes around blocks, and
researchers are told the same.

## Install

```sh
ownbox install odds
```

Manual:

```sh
git clone https://github.com/BogdanStamenovic/odds
cd odds
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/odds engines
```

You need at least one of `claude`, `codex`, or a running Ollama. The tool has
no runtime Python dependencies.

## Usage

```sh
odds ask "getting a one night stand in Korea as a foreign tourist"
odds ask --depth deep --research claude:sonnet,codex "how do I get upgraded to business class"
odds followup last "what about Busan instead of Seoul"
odds list
odds show last            # terminal summary;  --json for the raw run
odds report last --open   # re-render and open the HTML report
odds engines              # what is usable on this machine
```

| Option | Meaning |
|---|---|
| `--depth quick\|normal\|deep` | 4 / 7 / 12 subquestions; 0 / 1 / 2 critic rounds (default `normal`) |
| `--engine SPEC` | engine for the judgement calls (default `claude:opus`) |
| `--research SPECS` | comma-separated researcher engines, round-robin (default `claude:sonnet`) |
| `--sources NAMES` | restrict the lead adapters (default: all available) |
| `--no-ask` | never ask you anything; record assumptions instead |
| `--open` | open the HTML report when done |
| `-q`, `-v`, `--version` | the usual |

stdout carries only the result (summary, path, or JSON); progress and
questions go to stderr. Exit codes: `0` success, `1` the investigation failed
(the partial run is still saved), `2` usage error, `130` interrupted.

Runs live in `$ODDS_HOME/runs/` (default `~/.local/share/odds/runs/`), one
directory each, holding `run.json` and `report.html`.

## Limitations

- **The odds are estimates, not measurements.** They come from ranges an LLM
  read off the evidence. The Monte Carlo makes that uncertainty visible; it
  does not remove it. Treat a number as a screen ("roughly 1 in 5, not 1 in
  50"), not a verdict.
- **Stages are simulated as independent, and so are attempts.** In reality a
  good first impression lifts every later stage, and people learn between
  nights. Both simplifications are stated in `estimate.py`.
- **Researchers can still be wrong about what a page says.** Every sourced
  claim carries its URL, and the critic hunts for unsupported numbers, but
  nothing re-fetches pages to check quotes. Click the source before you bet on
  it.
- **The `claude -p` children inherit your global `~/.claude/CLAUDE.md`.** There
  is no keyless way to stop that (`--bare` drops the OAuth login). The prompt
  overrides it and the tool list is the actual fence. A child's own report of
  its tools is not evidence: in testing it claimed to have run a shell command
  it had no access to, and nothing ran.
- **Ollama is the weakest engine here.** With no web access it reasons only
  over scholarly and Wikipedia leads, and an 8B model's judgement shows. It
  works best as an extra researcher next to a web engine, not as the judge.
- **Reddit is unused until you register an app,** and its adapter has only been
  tested against canned payloads, never the live API.
- **Cost and time, measured** (same question, Opus judge, Sonnet researchers,
  2026-10-01). `cost` is what `claude -p` reports: on a subscription that is the
  notional API-price equivalent, not money charged.

  | depth | wall time | calls | sources | findings | critic | cost |
  |---|---|---|---|---|---|---|
  | quick | ~6 min | 7 | 39 | 40 (40 sourced, 0 inferred) | none | not tracked yet |
  | normal | 12.3 min | 17 (6 Opus, 11 Sonnet) | 73 | 111 (85 / 24 inferred / 2 speculation) | 11 problems, 4 new subquestions | $6.11 |

  Normal is slower than the ~10 minutes it was planned for. The researchers
  dominate: 1342 s of Sonnet time across 11 calls, run 6 at a time. `deep` has
  not been measured yet.
- **The quick run's tags were all "sourced".** With no critic and a looser rule
  at the time, researchers tagged everything as sourced. The rule is stricter
  now, but `quick` still has no critic to catch what slips through.

## License

MIT
