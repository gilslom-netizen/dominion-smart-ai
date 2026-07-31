//! Turning a self-played game into training examples.
//!
//! At every non-trivial decision, the search's own visit counts become the
//! policy target — this is the standard "distill the search into a network"
//! move: the network is not learning from game outcomes alone (a very weak
//! signal per decision), it is learning to reproduce a many-hundred-iteration
//! search using a single forward pass. The value target is filled in
//! afterwards, once the game's actual result is known.

use dominion_bots::policy;
use dominion_core::{Ctx, Game, Move, Rng};

use crate::evaluator::Evaluator;
use crate::example::Example;
use crate::features;
use crate::mcts::{search_with, MctsConfig};

/// Decisions where recording an example would not teach the network anything
/// a lookup table couldn't: `is_trivial` in `mcts.rs` skips searching them for
/// the same reason.
fn worth_recording(d: &dominion_core::Decision) -> bool {
    use Ctx::*;
    !matches!(d.ctx, MoatReveal)
}

/// Sampling temperature by turn: high early (so self-play games do not all
/// collapse to one line and the network sees a variety of positions), low
/// late (so the recorded value target reflects genuinely strong play).
fn temperature(turn: u32) -> f32 {
    if turn <= 6 {
        1.0
    } else {
        0.15
    }
}

fn sample_by_visits(rng: &mut Rng, visits: &[(Move, u32)], temperature: f32) -> Move {
    if temperature <= 0.0 {
        return visits.iter().max_by_key(|(_, v)| *v).unwrap().0;
    }
    let weights: Vec<f64> = visits
        .iter()
        .map(|&(_, v)| (v as f64 + 1.0).powf(1.0 / temperature as f64))
        .collect();
    let total: f64 = weights.iter().sum();
    let mut r = (rng.below(1_000_000) as f64 / 1_000_000.0) * total;
    for (w, &(mv, _)) in weights.iter().zip(visits) {
        if r < *w {
            return mv;
        }
        r -= w;
    }
    visits.last().unwrap().0
}

/// Play one self-play game to completion and return every recorded example,
/// value targets filled in.
pub fn play_selfplay_game(
    kingdom: &[dominion_core::Card],
    num_players: usize,
    seed: u64,
    cfg: &MctsConfig,
    eval: &dyn Evaluator,
    rng: &mut Rng,
) -> Vec<Example> {
    let mut game = Game::new(kingdom, num_players, seed).expect("valid kingdom");
    // (player, features, policy) — value filled in once the game ends.
    let mut pending: Vec<(usize, [f32; features::FEATURE_DIM], Vec<(Move, f32)>)> = Vec::new();

    let mut guard = 0u32;
    while !game.is_over() {
        let d = game.decision().expect("live game has a decision").clone();

        if d.options.len() == 1 {
            game.apply(d.options[0]).unwrap();
        } else if !worth_recording(&d) {
            let mv = policy::default_move(&game.state, &d);
            let mv = if d.options.contains(&mv) { mv } else { d.options[0] };
            game.apply(mv).unwrap();
        } else {
            let (_, visits) = search_with(&game.state, &d, cfg, eval, rng);
            let total: u32 = visits.iter().map(|(_, v)| v).sum::<u32>().max(1);
            let policy_target: Vec<(Move, f32)> = visits
                .iter()
                .map(|&(mv, v)| (mv, v as f32 / total as f32))
                .collect();

            let x = features::encode(&game.state, d.player, &d);
            pending.push((d.player, x, policy_target));

            let temp = temperature(game.state.players[d.player].turns);
            let mv = sample_by_visits(rng, &visits, temp);
            game.apply(mv).unwrap();
        }

        guard += 1;
        if guard > 2000 {
            break;
        }
    }

    let results = game.state.results();
    pending
        .into_iter()
        .map(|(player, features, policy)| Example {
            features,
            policy,
            value: results[player],
        })
        .collect()
}
