//! Answering "here is the game so far — what should I play?"
//!
//! The engine can rebuild any position from a log, and the search only ever
//! needs a state plus its pending decision. Put together, that means the AI can
//! be consulted at an arbitrary point in an arbitrary game without having
//! played it: replay the prefix, search the position, return the move.

use dominion_core::{Decision, GameLog, GameState, LogError, Move, Rng};

use crate::mcts::{search, MctsConfig};

/// A recommendation, with the search's reasoning attached.
#[derive(Clone, Debug)]
pub struct Advice {
    pub decision: Decision,
    pub best: Move,
    /// Visit counts across the ensemble, highest first. Their spread is the
    /// honest measure of how confident the search is.
    pub visits: Vec<(Move, u32)>,
    /// Whose turn number this is, for display.
    pub turn: u32,
}

impl Advice {
    /// Share of the ensemble's visits that went to the chosen move.
    pub fn confidence(&self) -> f32 {
        let total: u32 = self.visits.iter().map(|(_, v)| v).sum();
        if total == 0 {
            return 0.0;
        }
        self.visits
            .iter()
            .find(|(m, _)| *m == self.best)
            .map(|(_, v)| *v as f32 / total as f32)
            .unwrap_or(0.0)
    }
}

impl std::fmt::Display for Advice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "turn {} — player {} — {:?}",
            self.turn, self.decision.player, self.decision.ctx
        )?;
        writeln!(
            f,
            "recommended: {} ({:.0}% of visits)",
            self.best,
            self.confidence() * 100.0
        )?;
        writeln!(f, "considered:")?;
        for (mv, v) in self.visits.iter().take(8) {
            writeln!(f, "  {mv:<20} {v}")?;
        }
        Ok(())
    }
}

/// Search the position a state is currently parked on.
pub fn advise_state(state: &GameState, d: &Decision, cfg: &MctsConfig, rng: &mut Rng) -> Advice {
    let (best, mut visits) = search(state, d, cfg, rng);
    visits.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
    Advice {
        decision: d.clone(),
        best,
        visits,
        turn: state.players[d.player].turns,
    }
}

/// Replay `log` (optionally only its first `ply` moves) and advise on the
/// decision that follows.
pub fn advise_log(
    log: &GameLog,
    ply: Option<usize>,
    cfg: &MctsConfig,
    rng: &mut Rng,
) -> Result<Advice, LogError> {
    let game = match ply {
        Some(n) => log.replay_prefix(n)?,
        None => log.replay()?,
    };
    let Some(d) = game.decision().cloned() else {
        return Err(LogError::Parse(
            "the game is already over; there is nothing to decide".into(),
        ));
    };
    Ok(advise_state(&game.state, &d, cfg, rng))
}
