//! Random-playout fuzzing: every kingdom, every card, thousands of games.
//!
//! This is the engine's main safety net. A rules bug in Dominion usually shows
//! up as a panic (bad index, CardBuf overflow), a stuck game (decision loop
//! that never terminates) or a card-conservation violation (cards appearing
//! from or vanishing into nowhere).

use dominion_core::card::NUM_CARDS;
use dominion_core::{Card, Game, Move, Rng, KINGDOM_CARDS};

/// Total copies of each card across supply, trash and all player zones.
fn census(g: &Game) -> [u32; NUM_CARDS] {
    let mut c = [0u32; NUM_CARDS];
    for i in 0..NUM_CARDS {
        c[i] += g.state.supply[i] as u32;
    }
    for card in &g.state.trash {
        c[card.idx()] += 1;
    }
    for p in &g.state.players {
        for card in p.all_cards() {
            c[card.idx()] += 1;
        }
    }
    c
}

/// Play one game with uniformly random legal moves.
fn random_playout(kingdom: &[Card], seed: u64) -> Game {
    let mut g = Game::new(kingdom, 2, seed).expect("valid kingdom");
    let mut rng = Rng::new(seed ^ 0xA5A5_5A5A);
    let start = census(&g);
    let mut steps = 0u32;

    while !g.is_over() {
        let d = g.decision().expect("a live game always has a decision");
        assert!(
            d.options.len() > 1,
            "forced decisions should have been auto-resolved: {d:?}"
        );
        let mv = d.options[rng.below(d.options.len() as u64) as usize];
        g.apply(mv).expect("chose from the offered options");

        steps += 1;
        assert!(steps < 500_000, "game did not terminate");
    }

    assert_eq!(
        census(&g),
        start,
        "cards were created or destroyed during the game"
    );
    g
}

#[test]
fn random_games_on_random_kingdoms() {
    let mut rng = Rng::new(12345);
    for i in 0..400u64 {
        let kingdom = Game::random_kingdom(&mut rng);
        let g = random_playout(&kingdom, i);
        assert!(g.state.players.iter().all(|p| p.turns > 0));
    }
}

/// Every kingdom card gets exercised heavily, including the awkward ones, by
/// forcing each card into the kingdom in turn.
#[test]
fn every_card_gets_played() {
    let mut rng = Rng::new(777);
    for (i, &focus) in KINGDOM_CARDS.iter().enumerate() {
        for round in 0..12u64 {
            let mut pool: Vec<Card> = KINGDOM_CARDS.iter().copied().filter(|&c| c != focus).collect();
            rng.shuffle(&mut pool);
            let mut kingdom = vec![focus];
            kingdom.extend(pool.into_iter().take(9));
            kingdom.sort_unstable();
            random_playout(&kingdom, i as u64 * 1000 + round);
        }
    }
}

/// Two pure money players empty the Province pile in a plausible number of
/// turns. This pins down the economic backbone of the engine — draw, reshuffle,
/// coin counting and the end condition — independently of any kingdom card.
#[test]
fn money_only_game_ends_on_provinces() {
    let kingdom: Vec<Card> = KINGDOM_CARDS.iter().copied().take(10).collect();
    let mut g = Game::new(&kingdom, 2, 42).unwrap();
    while !g.is_over() {
        let d = g.decision().unwrap();
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
        g.apply(mv).unwrap();
    }
    assert_eq!(g.state.supply_of(Card::Province), 0);
    // Big Money mirror matches finish in the low-to-mid twenties.
    let turns = g.state.players[0].turns;
    assert!((15..40).contains(&turns), "unexpected game length: {turns}");
}
