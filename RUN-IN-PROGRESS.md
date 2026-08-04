# A self-play run is in progress

`selfplay-data/valuehead1-1785828834.gamelog` is being appended to right now.
An open run rewrites its file after every game, so it reports as modified on
every `git` command for the hours the run lasts. It has been checkpointed and
then marked assume-unchanged to stop that noise.

## Undo this when the run ends

```sh
git update-index --no-assume-unchanged selfplay-data/valuehead1-1785828834.gamelog
git add selfplay-data/valuehead1-1785828834.gamelog
```

**This is not optional bookkeeping.** While the flag is set, `git add` on that
path is a silent no-op: it succeeds, stages nothing, and every game after the
checkpoint is quietly dropped from all future commits. That already happened
once in this project — the flag was left set from a 280-game checkpoint, and
the completed 4,907-game run appeared to commit cleanly while adding nothing.
It was caught only because `git status` was read carefully afterwards.

Verify with `git ls-files -v selfplay-data/` — a lowercase `h` means the flag
is still set. `git add --dry-run` is the other reliable check. Do not trust a
clean-looking `git commit`.

## What this run is

The matched control for the rollout-leaf experiment: identical `--iterations
300`, identical `--net models/net.bin`, and an explicit `--seed` equal to the
tag-derived seed of the `rollout1` run, so both corpora cover the same
kingdoms and game seeds. The only difference is that this one prices search
leaves with the network's value head and `rollout1` prices them by rollout.

Without the matched seed and iteration count the comparison would be
confounded: every other corpus in `selfplay-data/` was generated at
`--iterations 1000`, so training against one of those would vary search depth
and leaf estimator together and credit the result to whichever was expected.
