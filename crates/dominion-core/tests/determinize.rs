//! Determinization must produce states that are (a) consistent with what the
//! observer knows and (b) actually playable, or search will explore positions
//! that cannot happen.

use dominion_core::determinize::{determinize, is_consistent_with};
use dominion_core::{Card, Game, Move, Rng};

/// Walk a game with random moves, determinizing at every decision point.
#[test]
fn determinized_states_stay_consistent_and_playable() {
    let mut rng = Rng::new(999);
    for round in 0..60u64 {
        let kingdom = Game::random_kingdom(&mut rng);
        let mut g = Game::new(&kingdom, 2, round).unwrap();
        let mut checked = 0;

        while !g.is_over() {
            let d = g.decision().unwrap().clone();

            for observer in 0..2 {
                let sampled = determinize(&g.state, observer, &mut rng);
                assert!(
                    is_consistent_with(&sampled, &g.state, observer),
                    "determinization changed something the observer can see"
                );
                for p in &sampled.players {
                    assert!(
                        p.known_top as usize <= p.deck.len(),
                        "known_top exceeds the deck it refers to"
                    );
                }
                checked += 1;
            }

            // A determinized state must be able to finish a legal game: the
            // pending decision survives and play continues from it.
            let mut branch = Game {
                state: determinize(&g.state, d.player, &mut rng),
            };
            assert_eq!(branch.decision().map(|x| x.ctx), Some(d.ctx));
            let mut steps = 0;
            while !branch.is_over() {
                let bd = branch.decision().unwrap();
                let mv = bd.options[rng.below(bd.options.len() as u64) as usize];
                branch.apply(mv).expect("determinized state plays legally");
                steps += 1;
                assert!(steps < 500_000, "determinized game did not terminate");
            }

            let mv = d.options[rng.below(d.options.len() as u64) as usize];
            g.apply(mv).unwrap();
        }
        assert!(checked > 20, "expected plenty of decision points to check");
    }
}

/// The observer's own hand is never resampled, and cards they deliberately
/// topdecked stay on top.
#[test]
fn a_topdecked_card_survives_determinization() {
    let kingdom = vec![
        Card::Artisan,
        Card::Cellar,
        Card::Chapel,
        Card::Moat,
        Card::Village,
        Card::Smithy,
        Card::Market,
        Card::Militia,
        Card::Mine,
        Card::Remodel,
    ];
    let mut g = Game::new(&kingdom, 2, 11).unwrap();
    {
        let pl = &mut g.state.players[0];
        pl.deck = vec![Card::Copper; 8];
        pl.deck.push(Card::Gold);
        pl.known_top = 1;
    }
    let mut rng = Rng::new(5);
    for _ in 0..50 {
        let s = determinize(&g.state, 0, &mut rng);
        assert_eq!(
            *s.players[0].deck.last().unwrap(),
            Card::Gold,
            "a known topdecked card must not be shuffled away"
        );
    }
}

/// Over many samples an opponent's hidden cards really do move around, so the
/// search sees genuine variety rather than one fixed guess.
#[test]
fn opponent_hidden_cards_get_resampled() {
    let mut rng = Rng::new(4242);
    let kingdom = Game::random_kingdom(&mut rng);
    let mut g = Game::new(&kingdom, 2, 77).unwrap();
    // Advance a few turns so the opponent's deck is non-trivial.
    for _ in 0..40 {
        if g.is_over() {
            break;
        }
        let d = g.decision().unwrap();
        let mv = d
            .options
            .iter()
            .find(|m| matches!(m, Move::Play(c) if c.is_treasure()))
            .copied()
            .unwrap_or(Move::Done);
        g.apply(mv).unwrap();
    }

    let mut seen = std::collections::HashSet::new();
    for _ in 0..100 {
        let s = determinize(&g.state, 0, &mut rng);
        seen.insert(s.players[1].hand.clone());
    }
    assert!(
        seen.len() > 5,
        "opponent's hand should vary across determinizations, saw {} variants",
        seen.len()
    );
}
