# Live self-play run — remove this file when it ends

`selfplay-data/rollout1-1785787810.gamelog` is being appended to by a running
`selfplay --rollout-leaves` process. It is committed at a checkpoint and then
marked:

```sh
git update-index --assume-unchanged selfplay-data/rollout1-1785787810.gamelog
```

so an hours-long run does not report the file as modified on every command.

**This must be undone when the run finishes**, or the remaining games are
silently left out of every future commit — which is exactly the kind of
quiet data loss the incremental-flush design exists to prevent:

```sh
git update-index --no-assume-unchanged selfplay-data/rollout1-1785787810.gamelog
git add selfplay-data/rollout1-1785787810.gamelog
git commit -m "rollout-leaf self-play, N games"
git push origin HEAD:main
rm RUN-IN-PROGRESS.md
```

Check whether the run is still alive with `pgrep -f "tag rollout1"`.
