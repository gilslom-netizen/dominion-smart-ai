//! Search-based agents.

pub mod mcts;
pub mod prior;

pub use mcts::{search, MctsAgent, MctsConfig};
