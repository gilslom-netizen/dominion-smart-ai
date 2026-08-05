// Rendering and input only. Every engine call happens in worker.js — see the
// note there for why the search cannot run on this thread.

const worker = new Worker('worker.js');
const $ = (id) => document.getElementById(id);

// Filled from the engine at boot: name, cost, types and rules text for all 33
// cards. Nothing about a card is written down here, so the page cannot drift
// from the implementation.
let CARDS = {};
const typeOf = (name) => {
  const c = CARDS[name];
  if (!c) return 'action';
  if (c.types.includes('Treasure')) return 'treasure';
  if (c.types.includes('Curse')) return 'curse';
  if (c.types.includes('Victory')) return 'victory';
  return 'action';
};

let busy = false;
let state = null;

// ---- card rendering -------------------------------------------------------

/** One card. `move` is the option index it would apply, or null if it is not
 *  a legal choice right now — those render dimmed rather than disappearing,
 *  so the board does not reshuffle itself between clicks. */
function card(name, n, move, extra = '') {
  const t = typeOf(name);
  const count = n > 1 ? `<span class="n">×${n}</span>` : '';
  const live = move !== null && move !== undefined;
  return `<button class="card t-${t} ${live ? 'live' : 'dead'}"
      data-card="${name}" ${live ? `data-move="${move}"` : ''}
      ${live ? '' : 'tabindex="-1"'}>
    <span class="cname">${name}</span>${count}${extra}
  </button>`;
}

/** Index of the option that plays/buys/picks this card, or null. */
function moveFor(kind, name) {
  if (!state || !state.options) return null;
  const i = state.options.findIndex((o) => o.kind === kind && o.card === name);
  return i === -1 ? null : i;
}

// ---- the board ------------------------------------------------------------

function render(s) {
  if (s.error) return;
  state = s;

  $('supply').innerHTML = s.supply
    .map((p) => {
      const mv = s.over ? null : moveFor('buy', p.card);
      const sum = CARDS[p.card]?.summary || '';
      const meta = `<span class="sum">${sum}</span><span class="meta">$${p.cost} · ${p.left} left</span>`;
      return card(p.card, 1, mv, meta).replace('class="card', `class="pile card${p.left === 0 ? ' gone' : ''}`);
    })
    .join('');
  $('empty-note').textContent =
    s.emptyPiles > 0 ? `${s.emptyPiles} pile${s.emptyPiles > 1 ? 's' : ''} empty — 3 ends the game` : '';

  // Hand: playable cards are clickable, and so are Select choices, which is
  // what most card effects ask for.
  $('hand').innerHTML =
    s.you.hand
      .map((c) =>
        card(
          c.card,
          c.n,
          s.over ? null : moveFor('play', c.card) ?? moveFor('pick', c.card),
          `<span class="sum">${CARDS[c.card]?.summary || ''}</span>`
        )
      )
      .join('') || '<span class="muted">empty</span>';

  const hasPlay = s.you.inPlay.length > 0;
  $('inplay-wrap').classList.toggle('hidden', !hasPlay);
  if (hasPlay) $('inplay').innerHTML = s.you.inPlay.map((c) => card(c.card, c.n, null)).join('');

  $('st-actions').textContent = s.you.actions;
  $('st-buys').textContent = s.you.buys;
  $('st-coins').textContent = '$' + s.you.coins;

  $('you-all').innerHTML = s.you.all.map((c) => card(c.card, c.n, null)).join('');
  $('ai-all').innerHTML = s.ai.all.map((c) => card(c.card, c.n, null)).join('');
  $('you-meta').textContent = `${s.you.cards} cards, ${s.you.vp} VP`;
  $('ai-meta').textContent = `${s.ai.cards} cards, ${s.ai.vp} VP`;

  $('ailog').innerHTML = s.aiLog.length
    ? s.aiLog.map((l) => `<li>${l}</li>`).join('')
    : '<li class="muted">nothing to show</li>';

  if (s.over) {
    $('prompt').textContent = 'Game over';
    $('otherwrap').classList.add('hidden');
    $('playall').classList.add('hidden');
    $('undo').disabled = true;
    const win = s.scores.you > s.scores.ai;
    const tie = s.scores.you === s.scores.ai;
    const el = $('result');
    el.className = 'result ' + (tie ? '' : win ? 'win' : 'lose');
    el.textContent = tie
      ? `Tie — ${s.scores.you} all.`
      : win
        ? `You win, ${s.scores.you} to ${s.scores.ai}.`
        : `The AI wins, ${s.scores.ai} to ${s.scores.you}.`;
    return;
  }

  $('result').className = 'result hidden';
  $('prompt').textContent = s.prompt;
  $('undo').disabled = !s.canUndo;
  $('playall').classList.toggle(
    'hidden',
    !s.options.some((o) => o.kind === 'play' && typeOf(o.card) === 'treasure')
  );

  // Anything not reachable by clicking a card on the board — "Done", and
  // gains from piles the supply row does not show — still needs a button.
  const shown = new Set();
  for (const el of document.querySelectorAll('.card[data-move]')) shown.add(Number(el.dataset.move));
  const leftover = s.options.map((o, i) => [o, i]).filter(([, i]) => !shown.has(i));
  $('otherwrap').classList.toggle('hidden', leftover.length === 0);
  $('options').innerHTML = leftover
    .map(([o, i]) => {
      const label = o.kind === 'done' ? 'Done' : o.label[0].toUpperCase() + o.label.slice(1);
      const cost = o.kind === 'buy' ? `<span class="cost">$${o.cost}</span>` : '';
      return `<button class="opt ${o.kind === 'done' ? 'done' : `t-${typeOf(o.card)}`}"
        data-move="${i}"><span>${label}</span>${cost}</button>`;
    })
    .join('');
}

// ---- input ----------------------------------------------------------------

function send(index) {
  if (busy || index === null || index === undefined) return;
  busy = true;
  worker.postMessage({ type: 'apply', index: Number(index) });
}

document.addEventListener('click', (ev) => {
  const el = ev.target.closest('[data-move]');
  if (el) {
    send(el.dataset.move);
    return;
  }
  if (!ev.target.closest('#cardinfo')) hideInfo();
});

// Right-click reads a card. Any card anywhere — hand, supply, either deck,
// the picker — since "what does this do" is the same question everywhere.
document.addEventListener('contextmenu', (ev) => {
  const el = ev.target.closest('[data-card]');
  if (!el) return;
  ev.preventDefault();
  showInfo(el.dataset.card, ev.clientX, ev.clientY);
});

function showInfo(name, x, y) {
  const c = CARDS[name];
  if (!c) return;
  $('ci-name').textContent = c.card;
  $('ci-cost').textContent = '$' + c.cost;
  $('ci-types').textContent = c.types;
  $('ci-text').textContent = c.text;
  const box = $('cardinfo');
  box.classList.remove('hidden');
  // Keep it on screen near the pointer rather than off the right edge.
  const w = box.offsetWidth;
  const h = box.offsetHeight;
  box.style.left = Math.min(x + 12, window.innerWidth - w - 8) + 'px';
  box.style.top = Math.min(y + 12, window.innerHeight - h - 8) + 'px';
}
const hideInfo = () => $('cardinfo').classList.add('hidden');
document.addEventListener('keydown', (e) => e.key === 'Escape' && hideInfo());
window.addEventListener('scroll', hideInfo, { passive: true });

$('undo').addEventListener('click', () => {
  if (busy) return;
  busy = true;
  worker.postMessage({ type: 'undo' });
});
$('playall').addEventListener('click', () => {
  if (busy) return;
  busy = true;
  worker.postMessage({ type: 'playAll' });
});

// ---- game log -------------------------------------------------------------

let logOpen = false;
$('togglelog').addEventListener('click', () => {
  logOpen = !logOpen;
  $('fulllog').classList.toggle('hidden', !logOpen);
  $('togglelog').textContent = logOpen ? 'Hide' : 'Show';
  if (logOpen) worker.postMessage({ type: 'history' });
});

// Games are kept without being asked for. There is no server here — that is
// the point of the wasm build — so a game only reaches anyone else if the
// player exports it, and waiting for someone to press Save before starting the
// next game would lose exactly the games worth having: the ones where
// something surprising happened and they immediately wanted another go.
const STORE = 'dominion-games';

function stored() {
  try {
    return JSON.parse(localStorage.getItem(STORE) || '[]');
  } catch {
    return [];
  }
}

function keep(text) {
  if (!text || !text.includes('moves:')) return;
  const games = stored();
  const seed = (text.match(/seed: (\d+)/) || [])[1] || '?';
  // Re-recording a seed replaces rather than duplicates, so undoing and
  // replaying the end of a game does not leave two versions of it.
  const i = games.findIndex((g) => g.seed === seed);
  const entry = { seed, at: new Date().toISOString(), text };
  if (i === -1) games.push(entry);
  else games[i] = entry;
  try {
    localStorage.setItem(STORE, JSON.stringify(games.slice(-50)));
  } catch {
    // Storage full or blocked: the Save button still works.
  }
  updateSaveLabel();
}

function updateSaveLabel() {
  const n = stored().length;
  $('savelog').textContent = n > 1 ? `Save ${n} games` : 'Save';
}

$('savelog').addEventListener('click', () => worker.postMessage({ type: 'logtext' }));

function saveLog(text) {
  keep(text);
  const games = stored();
  const body = games.length
    ? games.map((g) => `# game seed ${g.seed}, recorded ${g.at}\n${g.text}`).join('\n')
    : text;
  const blob = new Blob([body], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download =
    games.length > 1 ? `dominion-${games.length}-games.log` : `dominion-${games[0]?.seed || 'game'}.log`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---- worker ---------------------------------------------------------------

function fatal(message) {
  busy = true;
  $('thinking').classList.add('hidden');
  $('prompt').textContent = 'The engine failed to start';
  $('options').innerHTML = '';
  const el = $('result');
  el.className = 'result lose';
  el.textContent = message;
}
worker.onerror = (e) => fatal(e.message || 'the worker could not be loaded');

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'fatal') return fatal(m.message);
  if (m.type === 'cards') {
    for (const c of m.cards) CARDS[c.card] = c;
    renderPool(m.cards.filter((c) => c.kingdom));
    return;
  }
  if (m.type === 'history') {
    $('fulllog').innerHTML = m.history.length
      ? m.history.map((h) => `<li class="${h.you ? 'mine' : 'theirs'}">${h.label}</li>`).join('')
      : '<li class="muted">no moves yet</li>';
    return;
  }
  if (m.type === 'logtext') return saveLog(m.text);
  // Recorded silently at the end of every game, so nothing depends on the
  // player pressing anything.
  if (m.type === 'autolog') return keep(m.text);
  if (m.type === 'thinking') {
    $('thinking').classList.remove('hidden');
    return;
  }
  if (m.type === 'state') {
    busy = false;
    $('thinking').classList.add('hidden');
    render(m.state);
    if (logOpen) worker.postMessage({ type: 'history' });
    if (m.state.over) worker.postMessage({ type: 'autolog' });
  }
};

// ---- new game and the kingdom picker --------------------------------------

const picks = new Set();

function newGame() {
  const raw = $('seed').value.trim();
  const seed = raw === '' ? (Math.random() * 4294967295) >>> 0 : Number(raw) >>> 0;
  // Deliberately NOT written back into the input. Doing that turned the box
  // into a fixed seed after the first game, so every later "New game" replayed
  // the identical kingdom and opening hand.
  $('usedseed').textContent = 'seed ' + seed;
  busy = true;
  $('prompt').textContent = 'Dealing…';
  $('result').className = 'result hidden';
  worker.postMessage({
    type: 'new',
    seed,
    picks: [...picks],
    humanFirst: $('first').checked,
    worlds: 8,
    iterations: Number($('strength').value),
  });
}

function renderPool(pool) {
  $('pool').innerHTML = pool
    .map(
      (c) => `<label class="poolcard t-${typeOf(c.card)}" data-card="${c.card}">
        <input type="checkbox" data-i="${c.index}" ${picks.has(c.index) ? 'checked' : ''}>
        <span class="pname">${c.card}</span><span class="pcost">$${c.cost}</span>
      </label>`
    )
    .join('');
  for (const box of $('pool').querySelectorAll('input')) {
    box.addEventListener('change', () => {
      const i = Number(box.dataset.i);
      if (box.checked) {
        if (picks.size >= 10) {
          box.checked = false; // ten is the whole kingdom
          return;
        }
        picks.add(i);
      } else {
        picks.delete(i);
      }
      updatePickCount();
    });
  }
  updatePickCount();
}

function updatePickCount() {
  const n = picks.size;
  $('pickcount').textContent =
    n === 0
      ? '— none pinned, all 10 random'
      : n === 10
        ? '— all 10 chosen'
        : `— ${n} pinned, ${10 - n} random`;
}

$('clearpicks').addEventListener('click', (ev) => {
  ev.preventDefault();
  picks.clear();
  for (const box of $('pool').querySelectorAll('input')) box.checked = false;
  updatePickCount();
});

$('newgame').addEventListener('click', newGame);
worker.postMessage({ type: 'cards' });
updateSaveLabel();
newGame();
