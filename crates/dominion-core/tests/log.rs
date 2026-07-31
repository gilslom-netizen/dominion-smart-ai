//! A log plus a seed must reproduce a game exactly, and a prefix must
//! reproduce the position partway through it. Everything that consults the AI
//! about a game in progress rests on this.

use dominion_core::{Card, GameLog, Move, RecordedGame, Rng};
use dominion_core::engine::Game;

/// Play a game with a simple money policy, recording as we go.
fn recorded_money_game(seed: u64) -> RecordedGame {
    let kingdom = Game::random_kingdom(&mut Rng::new(seed));
    let mut rec = RecordedGame::new(&kingdom, 2, seed).unwrap();
    while !rec.game.is_over() {
        let d = rec.game.decision().unwrap().clone();
        let pick = |c: Card| d.options.iter().position(|m| *m == Move::Buy(c));
        let mv = if let Some(i) = d
            .options
            .iter()
            .position(|m| matches!(m, Move::Play(c) if c.is_treasure()))
        {
            d.options[i]
        } else if let Some(i) = pick(Card::Province)
            .or_else(|| pick(Card::Gold))
            .or_else(|| pick(Card::Silver))
        {
            d.options[i]
        } else {
            Move::Done
        };
        rec.apply(mv).unwrap();
    }
    rec
}

#[test]
fn replaying_a_log_reproduces_the_game_exactly() {
    for seed in 0..25u64 {
        let rec = recorded_money_game(seed);
        let replayed = rec.log.replay().expect("log replays");
        assert_eq!(
            replayed.state, rec.game.state,
            "replay diverged for seed {seed}"
        );
    }
}

#[test]
fn a_prefix_rebuilds_the_position_partway_through() {
    let rec = recorded_money_game(3);
    let total = rec.log.moves.len();
    assert!(total > 40, "expected a decent-length game, got {total} moves");

    // Walk a fresh game forward and check every prefix against it.
    let mut live = Game::new(&rec.log.kingdom, 2, rec.log.seed).unwrap();
    for n in 0..total {
        let from_prefix = rec.log.replay_prefix(n).expect("prefix replays");
        assert_eq!(from_prefix.state, live.state, "prefix {n} diverged");
        assert_eq!(
            from_prefix.decision().map(|d| d.ctx),
            live.decision().map(|d| d.ctx)
        );
        live.apply(rec.log.moves[n]).unwrap();
    }
}

#[test]
fn logs_survive_a_round_trip_through_text() {
    let rec = recorded_money_game(11);
    let text = rec.log.to_text();
    let parsed = GameLog::from_text(&text).expect("round trip parses");
    assert_eq!(parsed, rec.log);
    // And the parsed log still replays to the same finished game.
    assert_eq!(parsed.replay().unwrap().state, rec.game.state);
}

#[test]
fn a_log_that_does_not_match_the_rules_is_rejected() {
    let mut log = GameLog::new(
        Game::random_kingdom(&mut Rng::new(1)),
        2,
        5,
    );
    // Nobody can buy a Province on turn one.
    log.moves.push(Move::Buy(Card::Province));
    assert!(log.replay().is_err());
}

#[test]
fn text_format_is_readable_and_forgiving() {
    let text = "\
# a hand-written log
kingdom: Cellar, Chapel, Moat, Harbinger, Merchant, Vassal, Village, Workshop, Militia, Smithy
players: 2
seed: 99
moves:
  play Copper
  play Copper
";
    let log = GameLog::from_text(text).expect("hand-written logs parse");
    assert_eq!(log.seed, 99);
    assert_eq!(log.kingdom.len(), 10);
    assert_eq!(log.moves, vec![Move::Play(Card::Copper); 2]);
    assert!(log.replay().is_ok());
}
