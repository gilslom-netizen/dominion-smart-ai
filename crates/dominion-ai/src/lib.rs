//! Search-based agents, and the entry points for consulting them.

pub mod advise;
pub mod mcts;
pub mod prior;

pub use advise::{advise_log, advise_state, Advice};
pub use mcts::{search, MctsAgent, MctsConfig};
