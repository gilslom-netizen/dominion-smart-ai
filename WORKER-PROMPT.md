# Worker prompt — round 4 (measurement, not generation)

Copy everything below the line to an agent that has git access to
`https://github.com/gilslom-netizen/dominion-smart-ai`.

**Why this round is not more self-play.** An earlier draft of this file asked
every machine to regenerate self-play with `--rollout-leaves`, because that
leaf estimator measured much better and is worth +101 Elo when the search
plays. That instruction was withdrawn before being sent: two 4,900-game
corpora were generated over the same kingdoms and seeds, differing only in
the leaf estimator, and the networks trained on them are indistinguishable —
49.50% ± 2.89, 0.2σ. Better games do not make better training data here, and
they cost 4.3x to produce. Generating more of them would have burned four
machines for a day on a measured null.

What is actually scarce is **statistical power**. Nearly every result in this
project lands at ±2.5–4.5% on 120–500 games, which is wide enough that real
effects and noise look alike; two separate directions were chased on the
strength of results that later moved by 2σ when re-run larger. Parallel
machines fix exactly that, and they fix it without generating a single new
game.

---

You are helping measure a Dominion AI. Read `README.md` first — especially
"What did not work", which exists so ruled-out directions are not re-run.

## Setup

```sh
git clone https://github.com/gilslom-netizen/dominion-smart-ai
cd dominion-smart-ai
cargo build --release
cargo test --release          # should be 87 passing
```

## What to run

Pick the matchup below matching your agent name, and run it at a **large**
sample. These are the numbers currently too noisy to act on. Each takes a few
hours; report what you get, including the ± and the sigma.

```sh
# A — is the rollout default right at low budget too? (measured only at 8x400)
cargo run --release --example leaf_showdown -- models/net.bin 400 1

# B — the compute-matched arm at a third budget, to check it is not a
#     coincidence of the 8x400 timing ratio
cargo run --release --example leaf_showdown -- models/net.bin 400 2

# C — the best network against the heuristic and the menu ladder, large N
cargo run --release --bin bench -- 500

# D — re-measure the two corpora's networks at 1000 games rather than 300
#     (this is the null above; it deserves a tighter bound than 0.2σ at ±2.89)
cargo run --release --example net_vs_net -- models/net-a.bin models/net-b.bin 500
```

If your assigned letter's command needs a model file that is not in `models/`,
say so and stop rather than substituting a different one — a comparison
between the wrong two checkpoints is worse than no comparison.

## Reporting

Write `reports/<your-tag>-findings.md` with the raw output, not a summary of
it, and push:

```sh
git add reports/<your-tag>-findings.md
git commit -m "<tag>: measurement round 4"
git push origin HEAD:main       # HEAD:main, not main
```

## Rules

* **Report bugs, do not fix them.** Several agents patching in parallel is
  worse than the bugs.
* **Never overwrite `models/net.bin`** — the shared baseline. Do not run
  `train` this round at all.
* **`git push origin HEAD:main`**, not `git push origin main`. If your branch
  is not literally named `main`, the second form pushes a stale local ref and
  is rejected confusingly. This cost a previous agent real time.
* Report the number you measured even when it disagrees with the number in
  the README. That is the entire point of running it again.

## Known-good, so you do not re-report these

`selfplay --help` and `train --help` print usage and exit. Two agents last
round reported otherwise while running stale builds — rebuild before
reporting it. Unknown flags exit 2 rather than being silently ignored.
`bench` no longer extrapolates its self-play estimate from a one-sided game
and a linear-scaling assumption; it measures concurrent search-vs-search
games on the wall clock.
