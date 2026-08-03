# Agent report — tag `cwopus`

## Round 1 — trained network (worker prompt v1/v2)

- Machine: 4 logical cores, `estimated self-play rate: 0.5 games/s` (bench).
- Self-play: **2710 games** generated (`--iterations 300` default), stopped
  early — the measured rate (0.24 games/s, about half the bench estimate)
  put the requested 4500 games outside the 2-4h window, so the run was cut
  at 2710 to stay inside it. 648,578 training examples, 644,255 with
  distinct TD targets. Shard verified with `shard_stats` before training.
- Trained from the shared `models/net.bin` baseline, 8 epochs, pushed as
  `models/net-cwopus.bin`.
- Result: `NetMCTS(4x200) vs Heuristic 79.58% ± 3.68 (95W/24L/1D over 120
  games, +236 Elo, 19.9 turns, 5 games/s)`.

## Round 2 — deep-search self-play (worker prompt v3)

- Same machine, `--iterations 1000` (vs. default 300).
- First attempt: **180 games** generated before being told to stop — the
  `--seed` default of `0` meant every agent's run reproduced the identical
  game sequence regardless of `--tag`. Confirmed 100% pairwise overlap
  across three agents' files with `corpus_overlap`. Stopped, did not fix
  (per "report, don't fix"), waited for the seed-derivation fix to land on
  `main`.
- After the fix (`83862a6`, seed derived from `--tag`) landed and was
  pulled: re-verified 0% overlap with the fixed code, then restarted the
  run under a fresh 4-hour cap.
- **Final: 2370 games actually generated** (not the requested 100000 — that
  number was intentionally unreachable, per the prompt). The run stopped on
  its own 4-hour cap, not a crash or manual interrupt.
  File: `selfplay-data/cwopus-1785695414.gamelog`, 7.46 MB.
- Steady-state rate: 0.16 games/s for this run (the pre-fix 180-game run
  had measured 0.05-0.06 games/s under the same flags on the same machine;
  not investigated further, since diagnosing throughput swings wasn't
  asked for).

## Known leftover, not cleaned up

`selfplay-data/cwopus-1785691873.gamelog` (183 games, pre-seed-fix,
100% duplicate of `deep-verify` and `deep2` from the same period) is still
committed on `main`. Left in place pending an explicit decision — deleting
another agent's or a shared corpus file unilaterally seemed like the wrong
call to make without being asked.

## What was not done, on purpose

- No code changes at any point — only local verification (`shard_stats`,
  `corpus_overlap`, reading the arg parser) before running or pushing.
- `--iterations` was never changed from what each prompt specified.
- `train` was run only in round 1, as instructed there; not run in round 2.
- No `models/*.bin` pushed in round 2, no `models/net.bin` overwritten.
