//! Search-based agents, and the entry points for consulting them.

pub mod advise;
pub mod compact;
pub mod evaluator;
pub mod example;
pub mod features;
pub mod mcts;
pub mod net;
pub mod prior;
pub mod selfplay;

pub use advise::{advise_log, advise_state, Advice};
pub use evaluator::{BlendedEvaluator, Evaluator, HeuristicEvaluator, NetEvaluator};
pub use example::Example;
pub use features::{encode, FEATURE_DIM};
pub use mcts::{
    search, search_full, search_with, MctsAgent, MctsConfig, NetMctsAgent, SearchOutcome,
};
pub use net::{Net, OptConfig, Optimizer};
pub use selfplay::play_selfplay_game;
