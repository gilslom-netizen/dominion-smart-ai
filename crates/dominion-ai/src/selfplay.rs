//! Turning a self-played game into training examples.
//!
//! At every non-trivial decision, the search's own visit counts become the
//! policy target — this is the standard "distill the search into a network"
//! move: the network is not learning from game outcomes alone (a very weak
//! signal per decision), it is learning to reproduce a many-hundred-iteration
//! search using a single forward pass.
//!
//! For the **value** head the same logic applies, and it matters more here than
//! in most games. A Dominion game runs to roughly 240 decisions, so training
//! every one of them against the final win/loss is one bit of signal spread
//! impossibly thin: a buy on turn three is credited with an outcome it barely
//! influenced. That is textbook high-variance Monte Carlo credit assignment.
//!
//! So this records a TD(lambda) target instead, bootstrapping from the search's
//! own value estimates along the trajectory. Crucially the bootstrap comes from
//! **the search**, not from the raw network: the root value of a few thousand
//! ISMCTS simulations is a far better estimate than a single forward pass, so
//! the targets are useful even while the network itself is still poor. Both the
//! TD target and the raw outcome are stored, so the two can be compared without
//! regenerating data.

use dominion_bots::policy;
use dominion_core::{Ctx, Game, Move, Rng};

use crate::evaluator::Evaluator;
use crate::compact::{GameRecord, RecordedDecision};
use crate::example::Example;
use crate::features;
use crate::mcts::{search_full, MctsConfig};

/// Decisions where recording an example would not teach the network anything
/// a lookup table couldn't: `is_trivial` in `mcts.rs` skips searching them for
/// the same reason.
fn worth_recording(d: &dominion_core::Decision) -> bool {
    use Ctx::*;
    !matches!(d.ctx, MoatReveal)
}

/// TD(lambda) mixing weight. 1.0 is pure Monte Carlo (train on the final
/// outcome only); 0.0 is pure one-step bootstrap.
///
/// Dominion trajectories are long — around 240 decisions — so a value near 1
/// barely improves on Monte Carlo, while a value too low chases the search's
/// own noise. 0.9 gives an effective horizon of roughly ten decisions, which is
/// about how long a buy takes to show up in the deck.
pub const DEFAULT_LAMBDA: f32 = 0.9;

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
    play_selfplay_game_with_lambda(kingdom, num_players, seed, cfg, eval, rng, DEFAULT_LAMBDA)
}

/// As [`play_selfplay_game`], with an explicit TD(lambda) weight.
#[allow(clippy::too_many_arguments)]
pub fn play_selfplay_game_with_lambda(
    kingdom: &[dominion_core::Card],
    num_players: usize,
    seed: u64,
    cfg: &MctsConfig,
    eval: &dyn Evaluator,
    rng: &mut Rng,
    lambda: f32,
) -> Vec<Example> {
    let mut game = Game::new(kingdom, num_players, seed).expect("valid kingdom");
    // One entry per recorded decision; value targets are filled in afterwards,
    // backwards along the trajectory.
    struct Step {
        player: usize,
        features: [f32; features::FEATURE_DIM],
        policy: Vec<(Move, f32)>,
        /// The search's win-probability estimate for `player` at this point.
        search_value: f32,
    }
    let mut pending: Vec<Step> = Vec::new();

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
            let outcome = search_full(&game.state, &d, cfg, eval, rng);
            let total: u32 = outcome.visits.iter().map(|(_, v)| v).sum::<u32>().max(1);
            let policy_target: Vec<(Move, f32)> = outcome
                .visits
                .iter()
                .map(|&(mv, v)| (mv, v as f32 / total as f32))
                .collect();

            let x = features::encode(&game.state, d.player, &d);
            pending.push(Step {
                player: d.player,
                features: x,
                policy: policy_target,
                search_value: outcome.value,
            });

            let temp = temperature(game.state.players[d.player].turns);
            let mv = sample_by_visits(rng, &outcome.visits, temp);
            game.apply(mv).unwrap();
        }

        guard += 1;
        if guard > 2000 {
            break;
        }
    }

    let results = game.state.results();

    let trajectory: Vec<(usize, f32)> = pending
        .iter()
        .map(|s| (s.player, s.search_value))
        .collect();
    let targets = lambda_returns(&trajectory, &results, lambda);

    pending
        .into_iter()
        .zip(targets)
        .map(|(step, td_target)| Example {
            features: step.features,
            policy: step.policy,
            outcome: results[step.player],
            td_target,
        })
        .collect()
}

/// As [`play_selfplay_game_with_lambda`], but returning the compact
/// [`GameRecord`] as well — the form that is small enough to share between
/// machines.
#[allow(clippy::too_many_arguments)]
pub fn play_selfplay_game_recorded(
    kingdom: &[dominion_core::Card],
    num_players: usize,
    seed: u64,
    cfg: &MctsConfig,
    eval: &dyn Evaluator,
    rng: &mut Rng,
    lambda: f32,
) -> GameRecord {
    let mut game = Game::new(kingdom, num_players, seed).expect("valid kingdom");
    struct Step {
        player: usize,
        ply: u16,
        policy: Vec<(Move, f32)>,
        search_value: f32,
    }
    let mut pending: Vec<Step> = Vec::new();
    let mut moves: Vec<Move> = Vec::new();

    let mut guard = 0u32;
    while !game.is_over() {
        let d = game.decision().expect("live game has a decision").clone();
        let ply = moves.len() as u16;

        let mv = if !worth_recording(&d) {
            let m = policy::default_move(&game.state, &d);
            if d.options.contains(&m) { m } else { d.options[0] }
        } else {
            let outcome = search_full(&game.state, &d, cfg, eval, rng);
            let total: u32 = outcome.visits.iter().map(|(_, v)| v).sum::<u32>().max(1);
            pending.push(Step {
                player: d.player,
                ply,
                policy: outcome
                    .visits
                    .iter()
                    .map(|&(mv, v)| (mv, v as f32 / total as f32))
                    .collect(),
                search_value: outcome.value,
            });
            let temp = temperature(game.state.players[d.player].turns);
            sample_by_visits(rng, &outcome.visits, temp)
        };

        game.apply(mv).unwrap();
        moves.push(mv);

        guard += 1;
        if guard > 2000 {
            break;
        }
    }

    let results = game.state.results();
    let trajectory: Vec<(usize, f32)> = pending
        .iter()
        .map(|s| (s.player, s.search_value))
        .collect();
    let targets = lambda_returns(&trajectory, &results, lambda);

    GameRecord {
        kingdom: kingdom.to_vec(),
        players: num_players as u8,
        seed,
        moves,
        decisions: pending
            .into_iter()
            .zip(targets)
            .map(|(step, td_target)| RecordedDecision {
                ply: step.ply,
                policy: step.policy,
                outcome: results[step.player],
                td_target,
            })
            .collect(),
    }
}

/// TD(lambda) returns along one game's trajectory.
///
/// `trajectory` is `(deciding player, that player's searched win probability)`
/// per recorded decision, in play order. `results` is the final result per
/// player. Returns one target per decision, from its own player's perspective.
///
/// The recurrence, walked backwards with no intermediate rewards and no
/// discounting:
///
/// ```text
/// G_t = (1 - lambda) * V(s_{t+1}) + lambda * G_{t+1}
/// G_last = final result
/// ```
///
/// The one thing that is easy to get wrong: every quantity is expressed from
/// the perspective of whoever was deciding at *that* step, so when consecutive
/// steps belong to different players the next step's numbers must be flipped.
/// A position that is 0.8 for me is 0.2 for you.
pub fn lambda_returns(trajectory: &[(usize, f32)], results: &[f32], lambda: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; trajectory.len()];
    let mut next: Option<(usize, f32, f32)> = None; // (player, V, G)

    for (i, &(player, _)) in trajectory.iter().enumerate().rev() {
        let target = match next {
            None => results[player],
            Some((next_player, next_value, next_return)) => {
                let flip = |v: f32| if next_player == player { v } else { 1.0 - v };
                (1.0 - lambda) * flip(next_value) + lambda * flip(next_return)
            }
        };
        let target = target.clamp(0.0, 1.0);
        out[i] = target;
        next = Some((player, trajectory[i].1, target));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::HeuristicEvaluator;
    use dominion_core::Game;

    /// lambda = 1 is pure Monte Carlo: every step should carry the raw result.
    #[test]
    fn lambda_one_reduces_to_the_final_outcome() {
        let traj = vec![(0, 0.3), (1, 0.9), (0, 0.6), (1, 0.2)];
        let results = vec![1.0, 0.0];
        let g = lambda_returns(&traj, &results, 1.0);
        assert_eq!(g, vec![1.0, 0.0, 1.0, 0.0]);
    }

    /// lambda = 0 is a pure one-step bootstrap: each step takes the *next*
    /// step's searched value, flipped into its own perspective.
    #[test]
    fn lambda_zero_bootstraps_from_the_next_search_value() {
        //                   step 0        step 1
        let traj = vec![(0usize, 0.30), (1usize, 0.90)];
        let results = vec![1.0, 0.0];
        let g = lambda_returns(&traj, &results, 0.0);

        // Last step has nothing to bootstrap from: it takes the real result.
        assert!((g[1] - 0.0).abs() < 1e-6);
        // Step 0 belongs to player 0, step 1 to player 1. Player 1 rates their
        // own position 0.90, so from player 0's side that is 0.10.
        assert!((g[0] - 0.10).abs() < 1e-6, "got {}", g[0]);
    }

    /// Consecutive decisions by the same player must not be flipped.
    #[test]
    fn same_player_steps_are_not_flipped() {
        let traj = vec![(0usize, 0.30), (0usize, 0.80)];
        let results = vec![1.0, 0.0];
        let g = lambda_returns(&traj, &results, 0.0);
        assert!((g[1] - 1.0).abs() < 1e-6);
        assert!((g[0] - 0.80).abs() < 1e-6, "got {}", g[0]);
    }

    /// Intermediate lambda blends the two, and everything stays a probability.
    #[test]
    fn intermediate_lambda_blends_and_stays_in_range() {
        let traj = vec![(0usize, 0.30), (1usize, 0.90), (0usize, 0.55)];
        let results = vec![1.0, 0.0];
        let g = lambda_returns(&traj, &results, 0.5);
        assert!(g.iter().all(|v| (0.0..=1.0).contains(v)), "{g:?}");

        // Step 2 (player 0, last): the real result, 1.0.
        assert!((g[2] - 1.0).abs() < 1e-6);
        // Step 1 (player 1): next step is player 0 with V=0.55, G=1.0, so
        // flipped they are 0.45 and 0.0 -> 0.5*0.45 + 0.5*0.0 = 0.225.
        assert!((g[1] - 0.225).abs() < 1e-6, "got {}", g[1]);
        // Step 0 (player 0): next is player 1 with V=0.90, G=0.225, flipped
        // 0.10 and 0.775 -> 0.5*0.10 + 0.5*0.775 = 0.4375.
        assert!((g[0] - 0.4375).abs() < 1e-6, "got {}", g[0]);
    }

    #[test]
    fn an_empty_trajectory_is_handled() {
        assert!(lambda_returns(&[], &[1.0, 0.0], 0.9).is_empty());
    }

    /// End to end: a real self-play game produces sane targets, and lambda = 1
    /// makes the TD target collapse onto the outcome.
    #[test]
    fn a_real_game_produces_usable_targets() {
        let cfg = MctsConfig {
            worlds: 1,
            iterations: 15,
            ..Default::default()
        };
        let kingdom = Game::random_kingdom(&mut Rng::new(4));

        let mut rng = Rng::new(9);
        let td = play_selfplay_game_with_lambda(
            &kingdom, 2, 21, &cfg, &HeuristicEvaluator, &mut rng, 0.9,
        );
        assert!(td.len() > 20, "expected a real trajectory, got {}", td.len());
        for ex in &td {
            assert!((0.0..=1.0).contains(&ex.td_target), "{}", ex.td_target);
            assert!((0.0..=1.0).contains(&ex.outcome));
        }
        // Bootstrapping must actually change something, otherwise the whole
        // exercise is a no-op.
        assert!(
            td.iter().any(|e| (e.td_target - e.outcome).abs() > 1e-3),
            "TD targets are identical to the outcomes"
        );

        let mut rng = Rng::new(9);
        let mc = play_selfplay_game_with_lambda(
            &kingdom, 2, 21, &cfg, &HeuristicEvaluator, &mut rng, 1.0,
        );
        assert_eq!(mc.len(), td.len(), "lambda must not change the game played");
        for ex in &mc {
            assert!((ex.td_target - ex.outcome).abs() < 1e-6);
        }
    }
}
