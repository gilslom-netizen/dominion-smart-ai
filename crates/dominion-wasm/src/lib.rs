//! The engine and the search, compiled to WebAssembly so a browser can play
//! against the AI with no server at all.
//!
//! This is what makes the web version deployable as a static site: the search
//! runs on the player's machine, so there is no backend to pay for, no cold
//! start, no request that can time out mid-think, and no way for one player's
//! long search to slow down another's.
//!
//! # Interface
//!
//! Deliberately tiny — four calls and a JSON string — because a wasm boundary
//! is a place where mistakes are hard to debug from either side:
//!
//! * [`dom_new_game`] starts a game and returns which seat the human has.
//! * [`dom_state_json`] renders everything the UI draws, as JSON.
//! * [`dom_apply`] plays one of the moves the state listed.
//! * [`dom_ai_move`] searches and plays the AI's move.
//!
//! JSON crosses the boundary as UTF-8 in wasm memory; the caller reads
//! `dom_ptr()`/`dom_len()` and decodes. There is no allocator handshake, no
//! caller-owned buffer, and nothing for the JS side to free: the string lives
//! in a static owned by this module and is replaced on the next call.

use std::cell::RefCell;

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsAgent, MctsConfig, Net};
use dominion_bots::Agent;
use dominion_core::{Card, Ctx, Game, GameLog, GameState, Move, Rng, KINGDOM_CARDS};

/// The trained network, compiled in. ~130KB, which is small enough that
/// shipping it inside the module beats a second network request that can fail
/// on its own.
static NET_BYTES: &[u8] = include_bytes!("../../../models/net.bin");

struct Session {
    game: Game,
    /// Kept alongside the game because a `GameLog` needs them to replay, and
    /// the whole point of recording is that the log reconstructs the game.
    kingdom: Vec<Card>,
    seed: u64,
    human: usize,
    cfg: MctsConfig,
    net: Option<Net>,
    /// Moves played so far, so the UI can show a log and the position can be
    /// reconstructed exactly.
    history: Vec<(usize, Move)>,
    last_ai: Vec<String>,
    /// One snapshot per human decision, for undo.
    ///
    /// A whole `GameState` rather than a move list, because the engine is a
    /// continuation stack: replaying to a point means re-running every effect,
    /// while the state is a plain value that can be cloned at any decision.
    /// The cloned RNG is the point — restoring it means undoing and replaying
    /// the same move gives the same shuffle, so undo cannot be used to fish
    /// for a better draw.
    undo: Vec<(GameState, Vec<String>)>,
}

/// Kingdom cards the player pinned before starting, filled out at random.
thread_local! {
    static PICKS: RefCell<Vec<Card>> = const { RefCell::new(Vec::new()) };
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
    static OUT: RefCell<String> = const { RefCell::new(String::new()) };
}

fn put(s: String) {
    OUT.with(|o| *o.borrow_mut() = s);
}

/// Pointer to the current JSON string in wasm memory.
#[no_mangle]
pub extern "C" fn dom_ptr() -> *const u8 {
    OUT.with(|o| o.borrow().as_ptr())
}

/// Byte length of the current JSON string.
#[no_mangle]
pub extern "C" fn dom_len() -> usize {
    OUT.with(|o| o.borrow().len())
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn cards_json(cards: &[Card]) -> String {
    // Grouped and counted: a 45-card deck as 45 strings would be most of the
    // payload, and the UI wants counts anyway.
    let mut counts: Vec<(Card, usize)> = Vec::new();
    for c in cards {
        match counts.iter_mut().find(|(x, _)| x == c) {
            Some((_, n)) => *n += 1,
            None => counts.push((*c, 1)),
        }
    }
    counts.sort_by_key(|(c, _)| (c.cost(), format!("{c}")));
    let items: Vec<String> = counts
        .iter()
        .map(|(c, n)| format!("{{\"card\":\"{}\",\"n\":{}}}", esc(&format!("{c}")), n))
        .collect();
    format!("[{}]", items.join(","))
}

fn ctx_label(ctx: Ctx) -> &'static str {
    match ctx {
        Ctx::ActionPhase => "Play an Action, or end the phase",
        Ctx::BuyPhase => "Play a Treasure, buy a card, or end your turn",
        Ctx::MoatReveal => "Reveal Moat to block the attack?",
        Ctx::CellarDiscard => "Cellar: discard a card, then draw that many",
        Ctx::ChapelTrash => "Chapel: trash up to 4 cards",
        Ctx::HarbingerTopdeck => "Harbinger: put a card from your discard on top",
        Ctx::VassalPlay => "Vassal: play the discarded Action?",
        Ctx::WorkshopGain => "Workshop: gain a card costing up to $4",
        Ctx::BureaucratReveal => "Bureaucrat: reveal a Victory card to topdeck",
        Ctx::MilitiaDiscard => "Militia: discard down to 3 cards",
        Ctx::MoneylenderTrash => "Moneylender: trash a Copper for +$3?",
        Ctx::PoacherDiscard => "Poacher: discard one card per empty pile",
        Ctx::RemodelTrash => "Remodel: trash a card",
        Ctx::RemodelGain => "Remodel: gain a card costing up to $2 more",
        Ctx::ThroneRoomPlay => "Throne Room: choose an Action to play twice",
        Ctx::BanditTrash => "Bandit: choose which revealed Treasure is trashed",
        Ctx::LibrarySetAside => "Library: set this Action aside instead of keeping it?",
        Ctx::MineTrash => "Mine: trash a Treasure",
        Ctx::MineGain => "Mine: gain a Treasure costing up to $3 more",
        Ctx::SentryTrash => "Sentry: trash any of the revealed cards",
        Ctx::SentryDiscard => "Sentry: discard any of the rest",
        Ctx::SentryOrder => "Sentry: choose which card goes on top",
        Ctx::ArtisanGain => "Artisan: gain a card costing up to $5",
        Ctx::ArtisanTopdeck => "Artisan: put a card from hand on your deck",
    }
}

fn move_json(mv: &Move) -> String {
    let (kind, card) = match mv {
        Move::Play(c) => ("play", Some(*c)),
        Move::Buy(c) => ("buy", Some(*c)),
        Move::Select(c) => ("pick", Some(*c)),
        Move::Done => ("done", None),
    };
    match card {
        Some(c) => format!(
            "{{\"kind\":\"{kind}\",\"card\":\"{}\",\"cost\":{},\"label\":\"{}\"}}",
            esc(&format!("{c}")),
            c.cost(),
            esc(&format!("{mv}"))
        ),
        None => format!("{{\"kind\":\"{kind}\",\"card\":null,\"cost\":0,\"label\":\"Done\"}}"),
    }
}

fn state_json(s: &Session) -> String {
    let st = &s.game.state;
    let me = &st.players[s.human];
    let them = &st.players[1 - s.human];

    let mut supply = Vec::new();
    let mut empty = 0;
    for i in 0..dominion_core::NUM_CARDS {
        if !st.in_supply[i] {
            continue;
        }
        let card = Card::from_idx(i);
        let n = st.supply[i];
        if n == 0 {
            empty += 1;
        }
        supply.push(format!(
            "{{\"card\":\"{}\",\"cost\":{},\"left\":{}}}",
            esc(&format!("{card}")),
            card.cost(),
            n
        ));
    }

    let over = s.game.is_over();
    let scores = st.scores();
    let (options, ctx, whose) = if over {
        (String::from("[]"), String::new(), -1i32)
    } else {
        let d = s.game.decision().expect("live game has a decision");
        let opts: Vec<String> = d.options.iter().map(move_json).collect();
        (
            format!("[{}]", opts.join(",")),
            ctx_label(d.ctx).to_string(),
            d.player as i32,
        )
    };

    let log: Vec<String> = s
        .last_ai
        .iter()
        .map(|l| format!("\"{}\"", esc(l)))
        .collect();

    format!(
        "{{\"over\":{over},\"human\":{},\"toMove\":{whose},\"prompt\":\"{}\",\
         \"options\":{options},\"supply\":[{}],\"emptyPiles\":{empty},\
         \"you\":{{\"hand\":{},\"all\":{},\"inPlay\":{},\"actions\":{},\"buys\":{},\
         \"coins\":{},\"vp\":{},\"cards\":{}}},\
         \"ai\":{{\"all\":{},\"vp\":{},\"cards\":{}}},\
         \"scores\":{{\"you\":{},\"ai\":{}}},\"aiLog\":[{}],\"turns\":{}}}",
        s.human,
        esc(&ctx),
        supply.join(","),
        cards_json(&me.hand),
        cards_json(&me.all_cards().collect::<Vec<_>>()),
        cards_json(&me.play),
        me.actions,
        me.buys,
        me.coins,
        me.score(),
        me.total_cards(),
        cards_json(&them.all_cards().collect::<Vec<_>>()),
        them.score(),
        them.total_cards(),
        scores[s.human],
        scores[1 - s.human],
        log.join(","),
        s.history.len()
    )
}

/// Build the kingdom: every pinned card, then random ones up to ten.
///
/// Pinning fewer than ten is the useful case — "give me Witch and Chapel and
/// surprise me with the rest" — so the fill is random rather than the caller
/// having to choose all ten or none.
fn kingdom_from_picks(rng: &mut Rng) -> Vec<Card> {
    let picked: Vec<Card> = PICKS.with(|p| p.borrow().clone());
    let mut kingdom: Vec<Card> = Vec::new();
    for c in picked {
        if kingdom.len() < 10 && !kingdom.contains(&c) {
            kingdom.push(c);
        }
    }
    let mut pool: Vec<Card> = KINGDOM_CARDS
        .iter()
        .copied()
        .filter(|c| !kingdom.contains(c))
        .collect();
    rng.shuffle(&mut pool);
    for c in pool {
        if kingdom.len() == 10 {
            break;
        }
        kingdom.push(c);
    }
    kingdom.sort_unstable();
    kingdom
}

/// Forget every pinned card.
#[no_mangle]
pub extern "C" fn dom_clear_picks() {
    PICKS.with(|p| p.borrow_mut().clear());
}

/// Pin one kingdom card into the next game's supply, by card index. Ignores
/// anything that is not a kingdom card, and anything past ten.
#[no_mangle]
pub extern "C" fn dom_pick(card_index: u32) -> u32 {
    let i = card_index as usize;
    if i >= dominion_core::NUM_CARDS {
        return 0;
    }
    let card = Card::from_idx(i);
    if !KINGDOM_CARDS.contains(&card) {
        return 0;
    }
    PICKS.with(|p| {
        let mut b = p.borrow_mut();
        if b.len() < 10 && !b.contains(&card) {
            b.push(card);
        }
    });
    1
}

/// Every card, with cost, types and rules text, so the page never keeps its
/// own copy of any of it.
///
/// One call rather than one per card: the whole table is a few kilobytes and
/// never changes during a game, and a lookup per right-click would put a
/// worker round trip in front of a tooltip.
#[no_mangle]
pub extern "C" fn dom_cards_json() {
    let items: Vec<String> = dominion_core::ALL_CARDS
        .iter()
        .map(|c| {
            let mut kinds: Vec<&str> = Vec::new();
            if c.is_action() {
                kinds.push("Action");
            }
            if c.is_treasure() {
                kinds.push("Treasure");
            }
            if c.is_victory() {
                kinds.push("Victory");
            }
            if c.is_curse() {
                kinds.push("Curse");
            }
            if c.is_attack() {
                kinds.push("Attack");
            }
            if c.is_reaction() {
                kinds.push("Reaction");
            }
            format!(
                "{{\"index\":{},\"card\":\"{}\",\"cost\":{},\"kingdom\":{},\"types\":\"{}\",\"summary\":\"{}\",\"text\":\"{}\"}}",
                *c as usize,
                esc(&format!("{c}")),
                c.cost(),
                KINGDOM_CARDS.contains(c),
                kinds.join(" – "),
                esc(c.summary()),
                esc(c.text())
            )
        })
        .collect();
    put(format!("[{}]", items.join(",")));
}

/// Start a game. `human_first` of 1 puts the player on the first seat.
/// `strength` is search iterations per world; `worlds` is determinizations.
/// Returns 1 on success, 0 if the engine refused the parameters.
#[no_mangle]
pub extern "C" fn dom_new_game(seed: u32, human_first: u32, worlds: u32, iterations: u32) -> u32 {
    let seed = seed as u64;
    let mut krng = Rng::new(seed);
    let kingdom = kingdom_from_picks(&mut krng);
    let Ok(game) = Game::new(&kingdom, 2, seed) else {
        return 0;
    };
    let cfg = MctsConfig {
        worlds: worlds.max(1),
        iterations: iterations.max(1),
        ..Default::default()
    };
    let session = Session {
        game,
        kingdom: kingdom.clone(),
        seed,
        human: if human_first == 1 { 0 } else { 1 },
        cfg,
        net: Net::from_bytes(NET_BYTES),
        history: Vec::new(),
        last_ai: Vec::new(),
        undo: Vec::new(),
    };
    SESSION.with(|s| *s.borrow_mut() = Some(session));
    1
}

/// Render the current position as JSON into wasm memory.
#[no_mangle]
pub extern "C" fn dom_state_json() {
    let out = SESSION.with(|s| match s.borrow().as_ref() {
        Some(sess) => state_json(sess),
        None => "{\"error\":\"no game\"}".to_string(),
    });
    put(out);
}

/// Apply the option at `index` from the last state's `options`. Returns 1 if
/// it was applied.
///
/// The index is validated against the engine's own option list rather than
/// trusted, because the UI and the engine can drift apart — a stale click
/// after a re-render is an ordinary event, not an exceptional one.
#[no_mangle]
pub extern "C" fn dom_apply(index: u32) -> u32 {
    SESSION.with(|s| {
        let mut b = s.borrow_mut();
        let Some(sess) = b.as_mut() else { return 0 };
        if sess.game.is_over() {
            return 0;
        }
        let Some(d) = sess.game.decision().cloned() else {
            return 0;
        };
        let Some(&mv) = d.options.get(index as usize) else {
            return 0;
        };
        // Snapshot before the move, not after, so undo lands on the choice
        // rather than on its consequences.
        if d.player == sess.human {
            sess.undo.push((sess.game.state.clone(), sess.last_ai.clone()));
            if sess.undo.len() > 300 {
                sess.undo.remove(0);
            }
        }
        if sess.game.apply(mv).is_err() {
            sess.undo.pop();
            return 0;
        }
        sess.history.push((d.player, mv));
        1
    })
}

/// Take back the last decision, along with everything the AI did in reply.
///
/// Returns 1 if a move was taken back. Restoring the whole state includes the
/// RNG, so replaying the same move produces the same shuffle — undo cannot be
/// used to re-roll a bad draw, only to reconsider a choice.
#[no_mangle]
pub extern "C" fn dom_undo() -> u32 {
    SESSION.with(|s| {
        let mut b = s.borrow_mut();
        let Some(sess) = b.as_mut() else { return 0 };
        let Some((state, log)) = sess.undo.pop() else {
            return 0;
        };
        sess.game.state = state;
        sess.last_ai = log;
        sess.history.pop();
        1
    })
}

/// How many decisions can still be taken back.
#[no_mangle]
pub extern "C" fn dom_undo_depth() -> u32 {
    SESSION.with(|s| s.borrow().as_ref().map_or(0, |sess| sess.undo.len() as u32))
}

/// Play every Treasure in hand, as one action.
///
/// Playing Treasures one at a time is the single most repetitive thing in a
/// game of Dominion and it is never a real decision in the Base set: no
/// Treasure has a downside and nothing cares about unspent coins, which is
/// the same fact `prior::restrict` uses to collapse the choice for the search.
/// Returns how many were played.
#[no_mangle]
pub extern "C" fn dom_play_all_treasures() -> u32 {
    SESSION.with(|s| {
        let mut b = s.borrow_mut();
        let Some(sess) = b.as_mut() else { return 0 };
        let mut played = 0;
        let mut snapshotted = false;
        loop {
            if sess.game.is_over() {
                break;
            }
            let Some(d) = sess.game.decision().cloned() else {
                break;
            };
            if d.player != sess.human || d.ctx != Ctx::BuyPhase {
                break;
            }
            let Some(&mv) = d
                .options
                .iter()
                .find(|m| matches!(m, Move::Play(c) if c.is_treasure()))
            else {
                break;
            };
            // One snapshot for the whole batch: undo should take back "played
            // my Treasures", not one Copper of it.
            if !snapshotted {
                sess.undo.push((sess.game.state.clone(), sess.last_ai.clone()));
                snapshotted = true;
            }
            if sess.game.apply(mv).is_err() {
                break;
            }
            sess.history.push((d.player, mv));
            played += 1;
        }
        played
    })
}

/// Let the AI search and play one move. Returns 1 if it moved, 0 if it was not
/// its turn or the game is over.
#[no_mangle]
pub extern "C" fn dom_ai_move() -> u32 {
    SESSION.with(|s| {
        let mut b = s.borrow_mut();
        let Some(sess) = b.as_mut() else { return 0 };
        if sess.game.is_over() {
            return 0;
        }
        let Some(d) = sess.game.decision().cloned() else {
            return 0;
        };
        if d.player == sess.human {
            return 0;
        }
        let mv = match &sess.net {
            Some(n) => NetMctsAgent::new(sess.cfg, n).decide(&sess.game.state, &d),
            None => MctsAgent::new(sess.cfg).decide(&sess.game.state, &d),
        };
        if sess.game.apply(mv).is_err() {
            return 0;
        }
        // Only narrate what an opponent's move would reveal at a real table.
        if d.options.len() > 1 {
            sess.last_ai.push(format!("{mv}"));
            if sess.last_ai.len() > 40 {
                sess.last_ai.remove(0);
            }
        }
        sess.history.push((d.player, mv));
        1
    })
}

/// Clear the AI's move log, so the UI can show only what happened since the
/// player last acted.
#[no_mangle]
pub extern "C" fn dom_clear_log() {
    SESSION.with(|s| {
        if let Some(sess) = s.borrow_mut().as_mut() {
            sess.last_ai.clear();
        }
    });
}

/// Whether it is the human's turn to choose. The UI polls this instead of
/// reasoning about turn structure itself — the engine auto-resolves forced
/// moves, so "whose turn is it" is not something the client can derive.
#[no_mangle]
pub extern "C" fn dom_human_to_move() -> u32 {
    SESSION.with(|s| {
        let b = s.borrow();
        let Some(sess) = b.as_ref() else { return 0 };
        if sess.game.is_over() {
            return 0;
        }
        match sess.game.decision() {
            Some(d) if d.player == sess.human => 1,
            _ => 0,
        }
    })
}

/// The game so far in `GameLog` text form — the same format `bin/advise`
/// reads.
///
/// This is what makes a browser game worth keeping: the engine is
/// deterministic given a kingdom, a seed and a move list, so these few lines
/// reconstruct every position exactly. A game a human won can be replayed
/// afterwards and the AI asked, at any ply, what it would have done — which is
/// the only source of information about the AI's blind spots that self-play
/// cannot produce.
#[no_mangle]
pub extern "C" fn dom_log_text() {
    let out = SESSION.with(|s| match s.borrow().as_ref() {
        Some(sess) => {
            let mut log = GameLog::new(sess.kingdom.clone(), 2, sess.seed);
            log.moves = sess.history.iter().map(|(_, m)| *m).collect();
            log.to_text()
        }
        None => String::new(),
    });
    put(out);
}

/// Every move so far, with who made it, for the on-screen log.
#[no_mangle]
pub extern "C" fn dom_history_json() {
    let out = SESSION.with(|s| match s.borrow().as_ref() {
        Some(sess) => {
            let items: Vec<String> = sess
                .history
                .iter()
                .map(|(p, m)| {
                    format!(
                        "{{\"you\":{},\"label\":\"{}\"}}",
                        *p == sess.human,
                        esc(&format!("{m}"))
                    )
                })
                .collect();
            format!("[{}]", items.join(","))
        }
        None => "[]".to_string(),
    });
    put(out);
}

/// 1 once the game has finished.
#[no_mangle]
pub extern "C" fn dom_is_over() -> u32 {
    SESSION.with(|s| {
        let b = s.borrow();
        match b.as_ref() {
            Some(sess) if sess.game.is_over() => 1,
            _ => 0,
        }
    })
}
