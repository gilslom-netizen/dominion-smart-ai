#!/usr/bin/env bash
# Rebuild the WebAssembly module the site runs on.
#
# The committed web/public/dominion_wasm.wasm is a build artifact, kept in git
# so the site deploys as pure static files with no build step on Vercel's side.
# That means it goes stale silently whenever the engine, the search or
# models/net.bin changes — so run this after any of those.
set -euo pipefail
cd "$(dirname "$0")/.."

rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown -p dominion-wasm

cp target/wasm32-unknown-unknown/release/dominion_wasm.wasm web/public/
ls -la web/public/dominion_wasm.wasm
echo "rebuilt. serve locally with:  (cd web/public && python3 -m http.server 8000)"
