// Rendering only. Every engine call happens in worker.js — see the note there
// for why the search cannot run on this thread.

const worker = new Worker('worker.js');
const $ = (id) => document.getElementById(id);

// Card types, for the colour stripe. The engine knows these, but sending them
// per card would triple the payload for something that never changes, so the
// 33-card Base 2E pool is listed here instead.
const TREASURE = new Set(['Copper', 'Silver', 'Gold']);
const VICTORY = new Set(['Estate', 'Duchy', 'Province', 'Gardens']);
const CURSE = new Set(['Curse']);
const typeOf = (name) =>
  TREASURE.has(name) ? 'treasure' : VICTORY.has(name) ? 'victory' : CURSE.has(name) ? 'curse' : 'action';

let busy = false;

function chip(name, n, extra = '') {
  const t = typeOf(name);
  const count = n > 1 ? `<span>×${n}</span>` : '';
  return `<div class="chip t-${t}"><b>${name}</b>${count}${extra}</div>`;
}

function render(s) {
  if (s.error) return;

  // Supply
  $('supply').innerHTML = s.supply
    .map(
      (p) => `<div class="pile t-${typeOf(p.card)} ${p.left === 0 ? 'gone' : ''}">
        <div class="name">${p.card}</div>
        <div class="meta"><span>$${p.cost}</span><span>${p.left} left</span></div>
      </div>`
    )
    .join('');
  $('empty-note').textContent =
    s.emptyPiles > 0 ? `${s.emptyPiles} pile${s.emptyPiles > 1 ? 's' : ''} empty — 3 ends the game` : '';

  // Your side
  $('hand').innerHTML = s.you.hand.map((c) => chip(c.card, c.n)).join('') || '<span class="muted">empty</span>';
  const hasPlay = s.you.inPlay.length > 0;
  $('inplay-wrap').classList.toggle('hidden', !hasPlay);
  if (hasPlay) $('inplay').innerHTML = s.you.inPlay.map((c) => chip(c.card, c.n)).join('');

  $('st-actions').textContent = s.you.actions;
  $('st-buys').textContent = s.you.buys;
  $('st-coins').textContent = '$' + s.you.coins;

  $('you-all').innerHTML = s.you.all.map((c) => chip(c.card, c.n)).join('');
  $('ai-all').innerHTML = s.ai.all.map((c) => chip(c.card, c.n)).join('');
  $('you-meta').textContent = `${s.you.cards} cards, ${s.you.vp} VP`;
  $('ai-meta').textContent = `${s.ai.cards} cards, ${s.ai.vp} VP`;

  // What the AI did since you last acted
  $('ailog').innerHTML = s.aiLog.length
    ? s.aiLog.map((l) => `<li>${l}</li>`).join('')
    : '<li class="muted">nothing to show</li>';

  // Decision
  if (s.over) {
    $('prompt').textContent = 'Game over';
    $('options').innerHTML = '';
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
  const treasureToPlay = s.options.some((o) => o.kind === 'play' && TREASURE.has(o.card));
  $('playall').classList.toggle('hidden', !treasureToPlay);
  $('options').innerHTML = s.options
    .map((o, i) => {
      const cost = o.kind === 'buy' ? `<span class="cost">$${o.cost}</span>` : '';
      const label = o.kind === 'done' ? 'Done' : o.label[0].toUpperCase() + o.label.slice(1);
      return `<button class="opt ${o.kind === 'done' ? 'done' : `t-${typeOf(o.card)}`}" data-i="${i}">
        <span>${label}</span>${cost}</button>`;
    })
    .join('');

  for (const b of $('options').querySelectorAll('.opt')) {
    b.addEventListener('click', () => {
      if (busy) return;
      busy = true;
      worker.postMessage({ type: 'apply', index: Number(b.dataset.i) });
    });
  }
}

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
  if (e.data.type === 'fatal') {
    fatal(e.data.message);
    return;
  }
  if (e.data.type === 'pool') {
    renderPool(e.data.pool);
    return;
  }
  if (e.data.type === 'thinking') {
    $('thinking').classList.remove('hidden');
    return;
  }
  if (e.data.type === 'state') {
    busy = false;
    $('thinking').classList.add('hidden');
    render(e.data.state);
  }
};

function newGame() {
  const raw = $('seed').value.trim();
  const seed = raw === '' ? (Math.random() * 4294967295) >>> 0 : Number(raw) >>> 0;
  // Deliberately NOT written back into the input. Doing that turned the box
  // into a fixed seed after the first game, so every later "New game" replayed
  // the identical kingdom and opening hand. The seed used is shown separately,
  // read-only, so a game can still be reproduced on purpose.
  $('usedseed').textContent = 'seed ' + seed;
  busy = true;
  $('prompt').textContent = 'Dealing…';
  $('options').innerHTML = '';
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

// ---- kingdom picker -------------------------------------------------------

const picks = new Set();

function renderPool(pool) {
  $('pool').innerHTML = pool
    .map(
      (c) => `<label class="poolcard t-${typeOf(c.card)}">
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

// Hide the quick buttons while the game is over or the AI is thinking.
$('newgame').addEventListener('click', newGame);
worker.postMessage({ type: 'pool' });
newGame();
