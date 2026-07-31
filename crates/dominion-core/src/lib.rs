//! A rules engine for Dominion, Base set 2nd edition (26 kingdom cards).
//!
//! The engine is built for search and self-play rather than for a UI:
//!
//! * **Snapshot-anywhere.** Card effects run on an explicit continuation stack,
//!   so [`GameState`] is a plain value that can be cloned at any decision point.
//!   No coroutines, no callbacks into the agent.
//! * **Small, flat action space.** Compound choices are decomposed into
//!   single-card picks, giving a fixed [`MOVE_SPACE`] of 100 moves that a policy
//!   head can index directly.
//! * **Forced moves auto-resolve**, so callers only see choices that matter.

pub mod card;
pub mod determinize;
pub mod engine;
pub mod log;
pub mod rng;
pub mod state;

pub use card::{Card, CardCounts, Types, ALL_CARDS, BASIC_CARDS, KINGDOM_CARDS, NUM_CARDS};
pub use determinize::determinize;
pub use engine::{EngineError, Game};
pub use log::{GameLog, LogError, RecordedGame};
pub use rng::Rng;
pub use state::{
    Ctx, Decision, Dest, Frame, FrameKind, GameState, Move, PlayerState, MAX_PLAYERS, MOVE_SPACE,
};
