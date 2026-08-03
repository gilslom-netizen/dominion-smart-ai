//! What guides the search: a prior over moves, and — if available — a
//! shortcut past the rollout.
//!
//! Splitting this out as a trait is what lets [`crate::mcts`] stay unchanged
//! whether it is steered by the heuristic (as measured — see the README) or
//! by a trained [`crate::net::Net`]: the tree code only ever asks an
//! `Evaluator`, never `policy::default_move` or `Net` directly.

use dominion_core::{Decision, GameState, Move};

use crate::features;
use crate::net::Net;
use crate::prior;

pub trait Evaluator {
    /// A probability distribution over `options`, summing to 1.
    fn priors(&self, state: &GameState, d: &Decision, options: &[Move]) -> Vec<f32>;

    /// If this returns `Some`, the search uses it as the value of a freshly
    /// expanded leaf instead of playing a rollout out to the end of the game —
    /// the whole reason a value estimate is worth having. `player` is who is
    /// on the move at that leaf; the result is their win probability.
    ///
    /// The default is `None` everywhere, which makes the heuristic-only path
    /// behave exactly as before: always roll out.
    fn leaf_value(&self, _state: &GameState, _player: usize) -> Option<f32> {
        None
    }
}

/// The original heuristic prior. Identical behaviour to the search before
/// this module existed.
pub struct HeuristicEvaluator;

impl Evaluator for HeuristicEvaluator {
    fn priors(&self, state: &GameState, d: &Decision, options: &[Move]) -> Vec<f32> {
        prior::priors(state, d, options)
    }
}

/// A trained network, used for both the prior and the leaf value.
///
/// The value head is only meaningful for the player whose features it was
/// given, which is enough for 2-player games but not more — a 3+ player
/// leaf value would need a per-seat estimate the network was never trained
/// to produce, so it falls back to a rollout there.
pub struct NetEvaluator<'a> {
    pub net: &'a Net,
    /// Softening applied to the network's policy before it is used as a prior.
    ///
    /// 1.0 uses the network's distribution as-is. Higher values flatten it.
    ///
    /// This exists because the system was measured locking itself into a
    /// feedback loop: the search produced visit distributions with ~70% of the
    /// mass on one move, the network learned to reproduce that concentration,
    /// and PUCT's `c * P * sqrt(N) / (1 + n)` then spent every extra iteration
    /// reinforcing the same move rather than testing alternatives. Sixteen
    /// times the search budget changed the chosen move in 5.7% of decisions.
    /// Flattening the prior is the direct way to give the search room to
    /// disagree with it.
    pub temperature: f32,
}

impl<'a> NetEvaluator<'a> {
    /// The network's policy used exactly as trained.
    pub fn new(net: &'a Net) -> Self {
        NetEvaluator {
            net,
            temperature: 1.0,
        }
    }

    pub fn with_temperature(net: &'a Net, temperature: f32) -> Self {
        NetEvaluator { net, temperature }
    }
}

/// Raise a distribution to the power `1/t` and renormalise.
fn soften(p: &mut [f32], t: f32) {
    if (t - 1.0).abs() < 1e-6 || p.is_empty() {
        return;
    }
    let inv = 1.0 / t.max(1e-3);
    for v in p.iter_mut() {
        *v = v.max(1e-9).powf(inv);
    }
    let sum: f32 = p.iter().sum();
    if sum > 0.0 {
        for v in p.iter_mut() {
            *v /= sum;
        }
    }
}

impl<'a> Evaluator for NetEvaluator<'a> {
    fn priors(&self, state: &GameState, d: &Decision, options: &[Move]) -> Vec<f32> {
        let x = features::encode(state, d.player, d);
        let indices: Vec<usize> = options.iter().map(|m| m.index()).collect();
        let mut p = self.net.policy_over(&x, &indices);
        soften(&mut p, self.temperature);
        p
    }

    fn leaf_value(&self, state: &GameState, player: usize) -> Option<f32> {
        if state.num_players() != 2 {
            return None;
        }
        // The value head wants a Decision for the phase-indicator feature; a
        // leaf's actual pending decision context is a reasonable stand-in
        // since only the phase bit depends on it, not the legality of moves.
        let ctx = state
            .pending
            .as_ref()
            .map(|d| d.ctx)
            .unwrap_or(dominion_core::Ctx::BuyPhase);
        let d = Decision {
            player,
            ctx,
            options: Vec::new(),
        };
        let x = features::encode(state, player, &d);
        Some(self.net.value(&x))
    }
}

/// Blend of the heuristic prior with a network's — used while a network is
/// still early in training and should not be trusted alone, but is worth
/// nudging the search toward.
pub struct BlendedEvaluator<'a> {
    pub net: &'a Net,
    /// Weight given to the network's prior, in `[0, 1]`; the rest goes to the
    /// heuristic's.
    pub net_weight: f32,
}

impl<'a> Evaluator for BlendedEvaluator<'a> {
    fn priors(&self, state: &GameState, d: &Decision, options: &[Move]) -> Vec<f32> {
        let h = prior::priors(state, d, options);
        let x = features::encode(state, d.player, d);
        let indices: Vec<usize> = options.iter().map(|m| m.index()).collect();
        let n = self.net.policy_over(&x, &indices);
        h.iter()
            .zip(&n)
            .map(|(&hv, &nv)| (1.0 - self.net_weight) * hv + self.net_weight * nv)
            .collect()
    }
    // Leaf value intentionally left as the default `None`: a half-trained
    // network's value head is exactly what a rollout should be verifying
    // against during early self-play, not replacing.
}
