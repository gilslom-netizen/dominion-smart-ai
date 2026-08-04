# A self-play run is in progress

`selfplay-data/gen2-*.gamelog` is being appended to right now, so it reports
as modified on every git command until the run ends. It is checkpointed and
then marked `assume-unchanged` to stop that noise.

**Undo when the run finishes, before committing the full corpus:**

```sh
git update-index --no-assume-unchanged selfplay-data/gen2-*.gamelog
git ls-files -v selfplay-data/ | grep -v '^H'   # should print nothing
```

That flag is easy to leave set by accident, and while it is set `git add` on
that path is a silent no-op — every game after the checkpoint would be quietly
dropped from future commits, which is the same data loss the incremental-flush
design exists to prevent. It has already happened once in this project, which
is why the undo is written down rather than remembered.

This file should be deleted once no run is in progress.
