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

worker.onmessage = (e) => {
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
  $('seed').value = seed;
  busy = true;
  $('prompt').textContent = 'Dealing…';
  $('options').innerHTML = '';
  $('result').className = 'result hidden';
  worker.postMessage({
    type: 'new',
    seed,
    humanFirst: $('first').checked,
    worlds: 8,
    iterations: Number($('strength').value),
  });
}

$('newgame').addEventListener('click', newGame);
newGame();
