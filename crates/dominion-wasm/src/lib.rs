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
use dominion_core::{Card, Ctx, Game, Move, Rng};

/// The trained network, compiled in. ~130KB, which is small enough that
/// shipping it inside the module beats a second network request that can fail
/// on its own.
static NET_BYTES: &[u8] = include_bytes!("../../../models/net.bin");

struct Session {
    game: Game,
    human: usize,
    cfg: MctsConfig,
    net: Option<Net>,
    /// Moves played so far, so the UI can show a log and the position can be
    /// reconstructed exactly.
    history: Vec<(usize, Move)>,
    last_ai: Vec<String>,
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

/// Start a game. `human_first` of 1 puts the player on the first seat.
/// `strength` is search iterations per world; `worlds` is determinizations.
/// Returns 1 on success, 0 if the engine refused the parameters.
#[no_mangle]
pub extern "C" fn dom_new_game(seed: u32, human_first: u32, worlds: u32, iterations: u32) -> u32 {
    let seed = seed as u64;
    let mut krng = Rng::new(seed);
    let kingdom = Game::random_kingdom(&mut krng);
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
        human: if human_first == 1 { 0 } else { 1 },
        cfg,
        net: Net::from_bytes(NET_BYTES),
        history: Vec::new(),
        last_ai: Vec::new(),
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
        if sess.game.apply(mv).is_err() {
            return 0;
        }
        sess.history.push((d.player, mv));
        1
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
