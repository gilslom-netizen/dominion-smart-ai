// Owns the wasm module and every call into it.
//
// The search takes hundreds of milliseconds to several seconds per decision,
// which on the main thread would freeze the page — no scrolling, no hover, no
// repaint of the "thinking" indicator that exists to explain the wait. So the
// engine lives here and the page only ever renders.
//
// The worker also runs the AI's whole turn, not one move: the engine
// auto-resolves forced decisions, so "the AI's turn" is an unpredictable
// number of calls, and having the page drive that loop would put a message
// round trip between each of them.

let wasm = null;
let decoder = new TextDecoder();

async function boot() {
  const res = await fetch('dominion_wasm.wasm');
  const { instance } = await WebAssembly.instantiateStreaming(res, {});
  wasm = instance.exports;
}

const ready = boot();

function readState() {
  wasm.dom_state_json();
  const bytes = new Uint8Array(wasm.memory.buffer, wasm.dom_ptr(), wasm.dom_len());
  return JSON.parse(decoder.decode(bytes));
}

// Play AI moves until the human has a choice, or the game ends.
function runAiTurn() {
  let guard = 0;
  while (!wasm.dom_is_over() && !wasm.dom_human_to_move() && guard++ < 2000) {
    if (!wasm.dom_ai_move()) break;
  }
}

self.onmessage = async (e) => {
  await ready;
  const msg = e.data;

  if (msg.type === 'new') {
    wasm.dom_new_game(msg.seed >>> 0, msg.humanFirst ? 1 : 0, msg.worlds, msg.iterations);
    self.postMessage({ type: 'thinking' });
    runAiTurn();
    self.postMessage({ type: 'state', state: readState() });
    return;
  }

  if (msg.type === 'apply') {
    // A stale click — one sent against a board that has since changed — is
    // rejected by the engine, which validates the index itself. Re-render
    // rather than treating it as an error.
    if (!wasm.dom_apply(msg.index)) {
      self.postMessage({ type: 'state', state: readState() });
      return;
    }
    wasm.dom_clear_log();
    if (!wasm.dom_is_over() && !wasm.dom_human_to_move()) {
      self.postMessage({ type: 'thinking' });
    }
    runAiTurn();
    self.postMessage({ type: 'state', state: readState() });
    return;
  }
};
