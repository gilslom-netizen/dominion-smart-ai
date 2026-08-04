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

## If the page sits on "Dealing…"

That was a real failure mode and it can no longer happen silently. The module
is loaded with `instantiateStreaming`, which refuses any response not served
as `application/wasm` — and a static host sending `application/octet-stream`
for `.wasm` is common enough that `vercel.json` asking for the right type is
not sufficient. The worker now falls back to buffering and compiling the
bytes, so the MIME type no longer matters, and any remaining boot failure (a
404, a corrupt module) is reported on the page rather than leaving it
hanging.

## Caching

The module's filename carries a content hash
(`dominion_wasm.<hash>.wasm`), written by `build.sh` along with the
`engine-version.js` the worker reads it from. HTML, CSS and JS revalidate;
the hashed module is immutable, which is now safe because a rebuild produces
a different URL.

That indirection is not decoration — it fixes a bug that got shipped. The
module was first served under a fixed name with `immutable,
max-age=31536000`, so browsers *and the CDN* kept a year-old copy while the
HTML and JS updated normally. Fresh page code then called an export the old
module did not have, and the board never dealt. Only the production URL was
affected; a preview URL that had never cached anything worked, which is the
classic "works for me". A hard reload did not fix it either, because
Ctrl+Shift+R bypasses the browser cache and not the CDN. A new URL per build
is the only version of this that cannot go wrong.

The worker also verifies the module's exports at boot and names the missing
one, so if a stale module ever does reach it, the page says so instead of
hanging.

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
| `crates/dominion-wasm` | the C ABI: a handful of calls and a JSON string |
| `public/worker.js` | owns the wasm module; runs the search off the main thread |
| `public/app.js` | rendering and clicks only — no engine calls |
| `public/index.html`, `styles.css` | the board |

## Playing aids

* **Undo** restores a whole `GameState`, RNG included. Replaying the same move
  therefore produces the same shuffle, so undo lets you reconsider a choice
  but cannot be used to re-roll a bad draw. It takes back the AI's reply too,
  since taking back only your own move would leave you choosing against a
  position that already knows what you did.
* **Play all Treasures** is one click because it is never a real decision in
  the Base set: no Treasure has a downside and nothing cares about unspent
  coins. That is the same fact `prior::restrict` uses to collapse the choice
  for the search.
* **Choosing kingdom cards** pins as few or as many as you like and fills the
  rest at random, so "Witch and Chapel, surprise me with the other eight" is
  expressible. The card list comes from the engine rather than being copied
  into the page, so it cannot drift.

The worker exists because a search on the main thread freezes the page,
including the "thinking" indicator that is there to explain the wait. It also
plays the AI's *whole* turn rather than one move, since the engine
auto-resolves forced decisions and the number of calls in a turn is not
something the page can predict.
