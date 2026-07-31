//! Game logs: record a game as a replayable value, and rebuild the state at
//! any point in it.
//!
//! Because the engine is deterministic given a seed and a move sequence, a log
//! is just `(kingdom, players, seed, moves)` — no need to record the state.
//! Replaying a prefix rebuilds the exact position after those moves, which is
//! what "here is a game so far, what should I play?" needs.
//!
//! The text format is deliberately plain so logs can be written by hand, diffed
//! and pasted around.

use std::fmt::Write as _;

use crate::card::Card;
use crate::engine::{EngineError, Game};
use crate::state::Move;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GameLog {
    pub kingdom: Vec<Card>,
    pub players: usize,
    pub seed: u64,
    /// Every move answered, in order. Moves the engine auto-resolved are not
    /// recorded, because the engine never asked for them.
    pub moves: Vec<Move>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LogError {
    Engine(EngineError),
    Parse(String),
    /// A recorded move was not legal at that point in the replay.
    Diverged { at: usize, mv: Move },
}

impl std::fmt::Display for LogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogError::Engine(e) => write!(f, "{e}"),
            LogError::Parse(m) => write!(f, "parse error: {m}"),
            LogError::Diverged { at, mv } => {
                write!(f, "move {at} ({mv}) is not legal at that point")
            }
        }
    }
}

impl std::error::Error for LogError {}

impl GameLog {
    pub fn new(kingdom: Vec<Card>, players: usize, seed: u64) -> Self {
        GameLog {
            kingdom,
            players,
            seed,
            moves: Vec::new(),
        }
    }

    /// Rebuild the game after the first `n` recorded moves.
    ///
    /// The returned game is parked on the decision that follows them — ready to
    /// be searched, inspected, or continued.
    pub fn replay_prefix(&self, n: usize) -> Result<Game, LogError> {
        let mut game =
            Game::new(&self.kingdom, self.players, self.seed).map_err(LogError::Engine)?;
        for (i, &mv) in self.moves.iter().take(n).enumerate() {
            if game.is_over() {
                return Err(LogError::Diverged { at: i, mv });
            }
            game.apply(mv).map_err(|_| LogError::Diverged { at: i, mv })?;
        }
        Ok(game)
    }

    /// Rebuild the game at the end of the log.
    pub fn replay(&self) -> Result<Game, LogError> {
        self.replay_prefix(self.moves.len())
    }

    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let kingdom: Vec<&str> = self.kingdom.iter().map(|c| c.name()).collect();
        let _ = writeln!(s, "kingdom: {}", kingdom.join(", "));
        let _ = writeln!(s, "players: {}", self.players);
        let _ = writeln!(s, "seed: {}", self.seed);
        let _ = writeln!(s, "moves:");
        for mv in &self.moves {
            let _ = writeln!(s, "  {mv}");
        }
        s
    }

    pub fn from_text(text: &str) -> Result<GameLog, LogError> {
        let mut kingdom = Vec::new();
        let mut players = 2usize;
        let mut seed = 0u64;
        let mut moves = Vec::new();
        let mut in_moves = false;

        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let bad = |m: &str| LogError::Parse(format!("line {}: {m}", lineno + 1));

            if let Some(rest) = line.strip_prefix("kingdom:") {
                kingdom = rest
                    .split(',')
                    .map(|t| t.trim())
                    .filter(|t| !t.is_empty())
                    .map(|t| Card::parse(t).ok_or_else(|| bad(&format!("unknown card {t:?}"))))
                    .collect::<Result<_, _>>()?;
            } else if let Some(rest) = line.strip_prefix("players:") {
                players = rest.trim().parse().map_err(|_| bad("bad player count"))?;
            } else if let Some(rest) = line.strip_prefix("seed:") {
                seed = rest.trim().parse().map_err(|_| bad("bad seed"))?;
            } else if line.starts_with("moves:") {
                in_moves = true;
            } else if in_moves {
                moves.push(parse_move(line).ok_or_else(|| bad(&format!("bad move {line:?}")))?);
            } else {
                return Err(bad("unexpected line outside any section"));
            }
        }

        Ok(GameLog {
            kingdom,
            players,
            seed,
            moves,
        })
    }
}

/// Parse a move in the same form [`Move`] prints: `play Village`, `buy Silver`,
/// `pick Estate`, `done`.
pub fn parse_move(s: &str) -> Option<Move> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("done") {
        return Some(Move::Done);
    }
    let (verb, rest) = s.split_once(char::is_whitespace)?;
    let card = Card::parse(rest.trim())?;
    match verb.to_ascii_lowercase().as_str() {
        "play" => Some(Move::Play(card)),
        "buy" => Some(Move::Buy(card)),
        "pick" | "select" => Some(Move::Select(card)),
        _ => None,
    }
}

/// A game that records every move answered, so it can be replayed later.
pub struct RecordedGame {
    pub game: Game,
    pub log: GameLog,
}

impl RecordedGame {
    pub fn new(kingdom: &[Card], players: usize, seed: u64) -> Result<Self, EngineError> {
        Ok(RecordedGame {
            game: Game::new(kingdom, players, seed)?,
            log: GameLog::new(kingdom.to_vec(), players, seed),
        })
    }

    pub fn apply(&mut self, mv: Move) -> Result<(), EngineError> {
        self.game.apply(mv)?;
        self.log.moves.push(mv);
        Ok(())
    }
}
