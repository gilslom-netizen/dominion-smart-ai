# ccweb — findings from three self-play rounds

Agent tag: `ccweb`. Report only — no code was changed by this agent, per
worker-prompt instructions across all three rounds.

## Round 1 (shallow self-play + local training, `net.bin` baseline)

- Trained `models/net-ccweb.bin` from `models/net.bin`, 8 epochs, on
  **2,940 games** (heuristic-guided self-play, `--iterations 300`, the
  requested 4000 was not reached — run was time-boxed to 3h).
  - `NetMCTS(4x200) vs Heuristic  80.83% ± 3.59  (96W/22L/2D over 120 games, +250 Elo, 20.2 turns, 3 games/s)`
- **`--help` is not implemented on `selfplay`.** `cargo run --release --bin
  selfplay -- --help` silently launched a real run with default flags
  instead of printing usage. Caught it before any bad data was pushed;
  deleted the stray output. (Never independently confirmed the same for
  `train --help` — the one attempt to check it was chained after the
  hanging `selfplay --help` call and never actually ran.)
- **Seed collision hazard (now fixed upstream).** At the time, `--seed`
  defaulted to `0` and the per-game seed was derived only from `--seed` and
  the game index — so every agent that omitted `--seed` produced the exact
  same games. Worked around it locally by passing an explicit `--seed`.
  Confirmed fixed as of commit `5c2cd07`: seed now derives from `--tag`
  (`seed_from_tag`) unless `--seed` is passed explicitly.

## Round 2 (deep self-play, `--iterations 1000`, no training)

- First attempt (`selfplay-data/ccweb-1785691856.gamelog`, ~430 games,
  explicit manual `--seed 918273645`) was confirmed by manual overlap
  checking to be **100% duplicate** of another agent's corpus. Deleted, not
  pushed — no value beyond one copy, which existed elsewhere.
- Restarted after pulling the `seed_from_tag` fix, this time **omitting**
  `--seed` entirely as instructed. Ran to completion of a self-imposed 4h
  cap.
  - Final: **4,070 games**, `selfplay-data/ccweb-1785695440.gamelog`
    (12.9 MB, well under the 90 MB push limit).
  - Rate: ~0.12–0.13 games/s early, **0.28 games/s** average over the full
    4h run (`--iterations 1000` is measurably slower than the round-1
    default of 300, as expected — not treated as a bug).
  - Verified with `corpus_overlap` (once that example landed) that this
    file is not part of the earlier cross-agent duplication.
- **`--help` is still not implemented** as of commit `5c2cd07` (latest
  pulled at time of writing). Re-tested carefully in the background with a
  timeout after being told it was fixed — it still launches a real run
  (`generating 200 games ... -> selfplay-data/default-<ts>.gamelog
  (heuristic-guided)`) instead of printing usage. Caught and deleted the
  stray file both times (round 1 and round 2) before it could reach git.

## Net effect / recommendation

- `--help` should be fixed on `selfplay` (and `train` should be checked too)
  — right now typing it out of habit costs a real, uncommitted-but-wasted
  run every time.
- Everything else encountered this session (`.gitignore` blocking
  `.gamelog`, the seed-derivation bug) was already fixed upstream by the
  time it was checked — no outstanding action needed there.
