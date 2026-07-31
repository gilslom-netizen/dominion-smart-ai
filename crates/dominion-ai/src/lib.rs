//! Search-based agents, and the entry points for consulting them.

pub mod advise;
pub mod evaluator;
pub mod example;
pub mod features;
pub mod mcts;
pub mod net;
pub mod prior;
pub mod selfplay;

pub use advise::{advise_log, advise_state, Advice};
pub use features::{encode, FEATURE_DIM};
pub use evaluator::{BlendedEvaluator, Evaluator, HeuristicEvaluator, NetEvaluator};
pub use mcts::{search, search_with, MctsAgent, NetMctsAgent, MctsConfig};
pub use example::Example;
pub use net::Net;
pub use selfplay::play_selfplay_game;
