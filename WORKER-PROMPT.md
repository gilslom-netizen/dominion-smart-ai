# Worker prompt — round 4 (rollout-leaf self-play)

Copy everything below the line to an agent that has git access to
`https://github.com/gilslom-netizen/dominion-smart-ai`.

**What changed since round 3, and why this round exists:** the search was
measured running on the weaker of its two leaf estimates the entire time. The
network's value head correlates 0.547 with the game outcome; the heuristic
rollout it replaced correlates 0.639, and beats it in every third of the game
(Brier 0.1301 vs 0.1599 over 1,641 positions). Switching leaves is worth
64.17% ± 4.38 at equal search and 57.00% ± 2.21 over 500 games at equal wall
clock. Every game in `selfplay-data/` was generated before that fix, so all
existing training targets come from a handicapped search — which is the most
likely reason doubling the data kept measuring flat (49.25% ± 3.54).

---

You are contributing self-play data to a Dominion AI project. Read `README.md`
first — particularly "What did not work", which exists so directions already
ruled out are not re-run.

## What to do

```sh
git clone https://github.com/gilslom-netizen/dominion-smart-ai
cd dominion-smart-ai
cargo build --release

# Pick a short unique tag. It seeds generation, so two tags never produce the
# same games. Do NOT pass --seed.
export TAG=<your-agent-name>

cargo run --release --bin selfplay -- \
  --tag $TAG --games 100000 --net models/net.bin \
  --iterations 300 --rollout-leaves
```

`--rollout-leaves` is the point of this round. Do not omit it, and do not
change `--iterations`.

100000 is intentionally unreachable — it means "run until your time budget is
up". Give it 3–4 hours. The file is flushed after every game, so stopping at
any moment keeps everything finished so far. Then:

```sh
git add selfplay-data/$TAG-*.gamelog
git commit -m "rollout-leaf self-play from $TAG, N games"
git push origin HEAD:main       # HEAD:main, not main — see note below
```

## Expectations, so you can tell normal from broken

* **~4.3x slower than round 3 per game.** Rollouts play each leaf out instead
  of pricing it with one forward pass. A quarter the games in the same time is
  the expected outcome, not a problem to debug. Fewer, better games is the
  entire trade.
* Sanity-check your throughput against `cargo run --release --bin bench`, which
  now measures the self-play rate directly (one search-vs-search game per core,
  timed on the wall clock) rather than extrapolating. Earlier rounds reported
  its estimate as ~2x optimistic; that is fixed, but it measures the value-head
  path, so expect roughly a quarter of its number with `--rollout-leaves`.
* Before pushing, run `cargo run --release --example corpus_overlap --
  selfplay-data/*.gamelog` and confirm your file shows ~0% overlap with the
  others. A previous round had three agents produce byte-identical corpora.

## Rules

* **Report bugs, do not fix them.** Open an issue or write a findings file.
  Several agents patching the same code in parallel is worse than the bugs.
* **Never overwrite `models/net.bin`** — it is the shared baseline everyone
  starts from. Do not run `train` at all this round.
* **`git push origin HEAD:main`**, not `git push origin main`. If your assigned
  branch is not literally named `main`, the second form pushes a stale local
  `main` ref and is rejected confusingly. This cost a previous agent real time.
* Do not delete or modify another agent's file in `selfplay-data/`.

## Known-good checks

`selfplay --help` and `train --help` print usage and exit without starting a
run — two agents last round reported otherwise while running stale builds, so
if you see a real run start from `--help`, rebuild before reporting it.
Unknown flags exit 2 rather than being silently ignored, so a typo like
`--rollout-leafs` will tell you rather than quietly running the wrong
configuration for four hours.
