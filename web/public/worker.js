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
  if (!res.ok) throw new Error(`could not fetch the engine: HTTP ${res.status}`);

  // instantiateStreaming refuses anything not served as `application/wasm`,
  // and plenty of static hosts send `application/octet-stream` for .wasm
  // regardless of what the config asks for. That is not worth failing over:
  // buffer the bytes and compile them instead. Streaming stays the fast path
  // where the host cooperates.
  try {
    const { instance } = await WebAssembly.instantiateStreaming(res.clone(), {});
    wasm = instance.exports;
  } catch (streamErr) {
    const bytes = await res.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, {});
    wasm = instance.exports;
  }

  // A cached module from an older deploy is the failure this cannot afford to
  // meet quietly: the page's JS is fresh, the module is not, and the first
  // call to an export that did not exist yet throws somewhere the player only
  // sees as a board that never deals. Check up front and say so instead.
  const required = [
    'dom_new_game', 'dom_state_json', 'dom_apply', 'dom_ai_move',
    'dom_is_over', 'dom_human_to_move', 'dom_clear_log',
    'dom_undo', 'dom_undo_depth', 'dom_play_all_treasures',
    'dom_pick', 'dom_clear_picks', 'dom_pool_json', 'dom_ptr', 'dom_len',
  ];
  const missing = required.filter((n) => typeof wasm[n] !== 'function');
  if (missing.length) {
    throw new Error(
      `the cached engine is from an older version (missing ${missing[0]}). ` +
      `Reload with Ctrl+Shift+R to fetch the current one.`
    );
  }
}

// A boot failure used to leave the page sitting on "Dealing…" forever with
// nothing in the console the player would ever look at. Report it instead.
const ready = boot().catch((err) => {
  self.postMessage({ type: 'fatal', message: String(err && err.message ? err.message : err) });
  throw err;
});

function readJson() {
  const bytes = new Uint8Array(wasm.memory.buffer, wasm.dom_ptr(), wasm.dom_len());
  return JSON.parse(decoder.decode(bytes));
}

function readState() {
  wasm.dom_state_json();
  const s = readJson();
  s.canUndo = wasm.dom_undo_depth() > 0;
  return s;
}

// Play AI moves until the human has a choice, or the game ends.
function runAiTurn() {
  let guard = 0;
  while (!wasm.dom_is_over() && !wasm.dom_human_to_move() && guard++ < 2000) {
    if (!wasm.dom_ai_move()) break;
  }
}

function send() {
  self.postMessage({ type: 'state', state: readState() });
}

self.onmessage = async (e) => {
  await ready;
  const msg = e.data;

  if (msg.type === 'pool') {
    wasm.dom_pool_json();
    self.postMessage({ type: 'pool', pool: readJson() });
    return;
  }

  if (msg.type === 'new') {
    wasm.dom_clear_picks();
    for (const i of msg.picks || []) wasm.dom_pick(i);
    wasm.dom_new_game(msg.seed >>> 0, msg.humanFirst ? 1 : 0, msg.worlds, msg.iterations);
    self.postMessage({ type: 'thinking' });
    runAiTurn();
    send();
    return;
  }

  if (msg.type === 'apply') {
    // A stale click — one sent against a board that has since changed — is
    // rejected by the engine, which validates the index itself. Re-render
    // rather than treating it as an error.
    if (!wasm.dom_apply(msg.index)) {
      send();
      return;
    }
    wasm.dom_clear_log();
    if (!wasm.dom_is_over() && !wasm.dom_human_to_move()) {
      self.postMessage({ type: 'thinking' });
    }
    runAiTurn();
    send();
    return;
  }

  if (msg.type === 'playAll') {
    wasm.dom_play_all_treasures();
    send();
    return;
  }

  if (msg.type === 'undo') {
    // Undo lands on a human decision by construction, so the AI must not be
    // asked to move afterwards — doing so would immediately replay the turn
    // the player just took back.
    wasm.dom_undo();
    send();
    return;
  }
};
