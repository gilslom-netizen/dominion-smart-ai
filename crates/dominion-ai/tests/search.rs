//! Smoke tests for the search agent. Strength is measured by `bin/bench`;
//! these only check that search is well-formed and never illegal.

use dominion_ai::{prior, MctsAgent, MctsConfig};
use dominion_bots::buy::{big_money, MenuBot};
use dominion_bots::match_runner::play_game;
use dominion_bots::Agent;
use dominion_core::{Card, Ctx, Game, Move, Rng};

fn fast_cfg() -> MctsConfig {
    MctsConfig {
        worlds: 2,
        iterations: 30,
        ..Default::default()
    }
}

#[test]
fn search_only_ever_returns_legal_moves() {
    let mut rng = Rng::new(21);
    for round in 0..3u64 {
        let kingdom = Game::random_kingdom(&mut rng);
        let mut searcher = MctsAgent::new(fast_cfg());
        let mut foe = MenuBot::new(big_money());
        // play_game panics on an illegal move, which is the assertion here.
        let (results, turns) = play_game(&kingdom, &mut [&mut searcher, &mut foe], round);
        assert!((results.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(turns > 5, "game ended suspiciously early");
    }
}

#[test]
fn restriction_never_empties_the_move_set() {
    let mut rng = Rng::new(88);
    for round in 0..15u64 {
        let kingdom = Game::random_kingdom(&mut rng);
        let mut g = Game::new(&kingdom, 2, round).unwrap();
        while !g.is_over() {
            let d = g.decision().unwrap().clone();
            let opts = prior::restrict(&g.state, &d);
            assert!(!opts.is_empty(), "restriction removed every option");
            for m in &opts {
                assert!(d.options.contains(m), "restriction invented move {m}");
            }
            let p = prior::priors(&g.state, &d, &opts);
            assert_eq!(p.len(), opts.len());
            assert!(
                (p.iter().sum::<f32>() - 1.0).abs() < 1e-4,
                "priors must be a distribution, summed to {}",
                p.iter().sum::<f32>()
            );
            assert!(p.iter().all(|&x| x > 0.0), "priors must be positive");

            let mv = opts[rng.below(opts.len() as u64) as usize];
            g.apply(mv).unwrap();
        }
    }
}

/// Playing a Treasure is never a mistake in the Base set, and the order cannot
/// matter, so the buy phase should collapse to a single forced move until the
/// treasures are gone. This is what frees the search budget for real choices.
#[test]
fn treasure_plays_are_collapsed_to_one_option() {
    let kingdom = Game::random_kingdom(&mut Rng::new(2));
    let mut g = Game::new(&kingdom, 2, 4).unwrap();
    let mut saw_collapse = false;
    let mut steps = 0;
    while !g.is_over() && steps < 400 {
        let d = g.decision().unwrap().clone();
        if d.ctx == Ctx::BuyPhase {
            let opts = prior::restrict(&g.state, &d);
            let has_treasure = d
                .options
                .iter()
                .any(|m| matches!(m, Move::Play(c) if c.is_treasure()));
            if has_treasure {
                assert_eq!(opts.len(), 1, "treasure plays should collapse");
                assert!(matches!(opts[0], Move::Play(c) if c.is_treasure()));
                saw_collapse = true;
            }
        }
        let mv = prior::restrict(&g.state, &d)[0];
        g.apply(mv).unwrap();
        steps += 1;
    }
    assert!(saw_collapse, "never reached a buy phase holding a Treasure");
}

/// Curses are never worth buying in the Base set, so they should not reach the
/// search at all.
#[test]
fn buying_a_curse_is_never_offered_to_the_search() {
    let mut rng = Rng::new(5);
    let kingdom = Game::random_kingdom(&mut rng);
    let mut g = Game::new(&kingdom, 2, 9).unwrap();
    while !g.is_over() {
        let d = g.decision().unwrap().clone();
        let opts = prior::restrict(&g.state, &d);
        assert!(!opts.contains(&Move::Buy(Card::Curse)));
        let mv = opts[rng.below(opts.len() as u64) as usize];
        g.apply(mv).unwrap();
    }
}
