//! Reference agents.
//!
//! Two things live here, and they are deliberately separate:
//!
//! * [`policy`] — a decent default answer for every non-buy decision in the
//!   game (which Action to play, what to discard to Militia, what to trash to
//!   Chapel). Every bot shares it, so strategies differ only where they mean to.
//! * [`buy`] — declarative buy menus in the style of Geronimoo's simulator.
//!   This is where a named strategy such as Big Money actually lives.
//!
//! The heuristic layer doubles as the rollout policy for search: random
//! rollouts in Dominion are nearly worthless because a random buy phase never
//! builds an economy, so playouts need a baseline that at least buys money.

pub mod buy;
pub mod match_runner;
pub mod policy;

use dominion_core::{Decision, GameState, Move};

/// Something that can answer decisions.
///
/// Agents are handed the full [`GameState`]. Heuristic agents here restrict
/// themselves to information their player could legitimately know; search
/// agents will use the observation layer to enforce that structurally.
pub trait Agent {
    fn decide(&mut self, state: &GameState, decision: &Decision) -> Move;
    fn name(&self) -> String;
    /// Called once at the start of each game so stateful agents can reset.
    fn reset(&mut self) {}
}

pub use buy::{BuyMenu, MenuBot};
pub use match_runner::{play_game, run_match, run_match_parallel, Kingdoms, MatchResult};
pub use policy::{DeckStats, HeuristicBot, RandomAgent};
