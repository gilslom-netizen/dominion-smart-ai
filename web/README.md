# The web version

Play the AI in a browser: <https://vercel.com> serves this directory as a
static site, and the engine plus the search run entirely on the visitor's
machine as WebAssembly.

## Why no server

The search costs hundreds of milliseconds to a few seconds per decision. Run
server-side that is a per-move request that can time out, a per-visitor cost,
and a queue where one player's long think delays another's. Compiled to wasm
it is none of those: no backend, no cold start, no rate limit, and the
"stronger but slower" setting costs the player their own CPU rather than
ours.

The trade is a 335KB module, downloaded once and cached immutably, which
includes the trained network (`models/net.bin`) compiled in.

## Deploying

The site is `web/public` — plain static files, no build step.

```sh
npx vercel --cwd web          # preview
npx vercel --cwd web --prod   # production
```

Or point a Vercel project at this repo with **root directory** `web` and no
build command. `web/vercel.json` sets the wasm content type and cache header.

## Developing

```sh
./web/build.sh                              # rebuild the wasm module
(cd web/public && python3 -m http.server 8000)
```

`web/public/dominion_wasm.wasm` is a committed build artifact — that is what
keeps the deploy static. It therefore goes stale silently whenever the engine,
the search or the network changes, so `build.sh` is not optional after those.

## How it fits together

| file | role |
|---|---|
| `crates/dominion-wasm` | the C ABI: four calls and a JSON string |
| `public/worker.js` | owns the wasm module; runs the search off the main thread |
| `public/app.js` | rendering and clicks only — no engine calls |
| `public/index.html`, `styles.css` | the board |

The worker exists because a search on the main thread freezes the page,
including the "thinking" indicator that is there to explain the wait. It also
plays the AI's *whole* turn rather than one move, since the engine
auto-resolves forced decisions and the number of calls in a turn is not
something the page can predict.
