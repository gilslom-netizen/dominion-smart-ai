//! A self-play format small enough to travel between machines.
//!
//! The `.shard` format stores a 139-float feature vector per decision, which is
//! 93% of every record and puts a 3000-game run at 427MB — far past what GitHub
//! will accept. So the machines generating self-play in parallel could not
//! actually pool their data, and combining their *networks* instead turned out
//! not to work (see `Net::weighted_average`): four contributors totalling
//! 12,230 games averaged into something that could not beat the single
//! 3,500-game one.
//!
//! But the features are derivable. The engine is deterministic given a kingdom,
//! a seed and a move sequence, and `dominion_core::GameLog` already replays
//! exactly. So this format stores the *game* and re-derives the features at
//! training time:
//!
//! ```text
//! per decision:  ply(u16) + n(u8) + n*(move u8, prob u16) + outcome u8 + td u16
//! per game:      kingdom + seed + moves + the decisions above
//! ```
//!
//! 30 bytes per decision instead of 600, and 23MB per 3000 games instead of
//! 427MB. Replaying to expand it costs a few seconds for thousands of games —
//! the engine runs several thousand full games a second — which is nothing
//! against the hours the games took to search in the first place.
//!
//! Probabilities are stored as `u16` fixed point. A visit distribution is a
//! training target, not an exact quantity, and 1.5e-5 resolution is far finer
//! than the sampling noise in the visit counts it came from.

use dominion_core::state::MOVE_SPACE;
use dominion_core::{Card, Game, Move};

use crate::example::Example;
use crate::features;

const MAGIC: u32 = 0xD0A1_106C;

/// One decision worth keeping out of a replayed game.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordedDecision {
    /// Which decision of the game this was, counting from zero. Stored rather
    /// than recomputed so that changing which decisions are worth recording
    /// cannot silently misalign old data against new code.
    pub ply: u16,
    pub policy: Vec<(Move, f32)>,
    pub outcome: f32,
    pub td_target: f32,
}

/// A whole self-play game, compressed to what is needed to rebuild it.
#[derive(Clone, Debug, PartialEq)]
pub struct GameRecord {
    pub kingdom: Vec<Card>,
    pub players: u8,
    pub seed: u64,
    /// Every move applied, in order — enough to replay the game exactly.
    pub moves: Vec<Move>,
    pub decisions: Vec<RecordedDecision>,
}

#[derive(Debug)]
pub enum CompactError {
    NotACompactFile,
    Truncated,
    Malformed(String),
    /// Replay reached a different position than the one recorded. This means
    /// the data and the engine disagree, and expanding it would produce
    /// plausible-looking but wrong training features.
    ReplayDiverged { game: usize, detail: String },
}

impl std::fmt::Display for CompactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompactError::NotACompactFile => write!(f, "not a compact self-play file"),
            CompactError::Truncated => write!(f, "file ends mid-record"),
            CompactError::Malformed(m) => write!(f, "malformed record: {m}"),
            CompactError::ReplayDiverged { game, detail } => write!(
                f,
                "game {game} did not replay as recorded ({detail}) — \
                 the engine and this data disagree, refusing to train on it"
            ),
        }
    }
}

impl std::error::Error for CompactError {}

fn p_to_u16(p: f32) -> u16 {
    (p.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16
}
fn u16_to_p(v: u16) -> f32 {
    v as f32 / u16::MAX as f32
}

// ------------------------------------------------------------------ writing

/// Append games to a compact file, creating it if needed.
///
/// Like the shard format, this is append-only and safe to call repeatedly, so
/// a long run can flush as it goes and survive being interrupted.
pub fn append_games(path: &str, games: &[GameRecord]) -> std::io::Result<()> {
    use std::io::Write;
    let is_new = !std::path::Path::new(path).exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if is_new {
        f.write_all(&MAGIC.to_le_bytes())?;
    }
    let mut buf = Vec::new();
    for g in games {
        encode_game(g, &mut buf);
    }
    f.write_all(&buf)
}

fn encode_game(g: &GameRecord, out: &mut Vec<u8>) {
    out.push(g.kingdom.len() as u8);
    for c in &g.kingdom {
        out.push(*c as u8);
    }
    out.push(g.players);
    out.extend_from_slice(&g.seed.to_le_bytes());

    out.extend_from_slice(&(g.moves.len() as u32).to_le_bytes());
    for m in &g.moves {
        out.push(m.index() as u8);
    }

    out.extend_from_slice(&(g.decisions.len() as u32).to_le_bytes());
    for d in &g.decisions {
        out.extend_from_slice(&d.ply.to_le_bytes());
        out.push(d.policy.len() as u8);
        for (mv, p) in &d.policy {
            out.push(mv.index() as u8);
            out.extend_from_slice(&p_to_u16(*p).to_le_bytes());
        }
        out.push((d.outcome.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.extend_from_slice(&p_to_u16(d.td_target).to_le_bytes());
    }
}

// ------------------------------------------------------------------ reading

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.p)?;
        self.p += 1;
        Some(v)
    }
    fn u16(&mut self) -> Option<u16> {
        let c = self.b.get(self.p..self.p + 2)?;
        self.p += 2;
        Some(u16::from_le_bytes(c.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        let c = self.b.get(self.p..self.p + 4)?;
        self.p += 4;
        Some(u32::from_le_bytes(c.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        let c = self.b.get(self.p..self.p + 8)?;
        self.p += 8;
        Some(u64::from_le_bytes(c.try_into().ok()?))
    }
}

fn decode_game(c: &mut Cursor) -> Option<GameRecord> {
    let n_kingdom = c.u8()? as usize;
    if n_kingdom > dominion_core::NUM_CARDS {
        return None;
    }
    let mut kingdom = Vec::with_capacity(n_kingdom);
    for _ in 0..n_kingdom {
        let idx = c.u8()? as usize;
        if idx >= dominion_core::NUM_CARDS {
            return None;
        }
        kingdom.push(Card::from_idx(idx));
    }
    let players = c.u8()?;
    let seed = c.u64()?;

    let n_moves = c.u32()? as usize;
    // A truncated length must not drive a huge allocation.
    if n_moves > 100_000 {
        return None;
    }
    let mut moves = Vec::with_capacity(n_moves);
    for _ in 0..n_moves {
        moves.push(Move::from_index(c.u8()? as usize)?);
    }

    let n_dec = c.u32()? as usize;
    if n_dec > 100_000 {
        return None;
    }
    let mut decisions = Vec::with_capacity(n_dec);
    for _ in 0..n_dec {
        let ply = c.u16()?;
        let n_pol = c.u8()? as usize;
        if n_pol > MOVE_SPACE {
            return None;
        }
        let mut policy = Vec::with_capacity(n_pol);
        for _ in 0..n_pol {
            let mv = Move::from_index(c.u8()? as usize)?;
            policy.push((mv, u16_to_p(c.u16()?)));
        }
        let outcome = c.u8()? as f32 / 255.0;
        let td_target = u16_to_p(c.u16()?);
        decisions.push(RecordedDecision {
            ply,
            policy,
            outcome,
            td_target,
        });
    }

    Some(GameRecord {
        kingdom,
        players,
        seed,
        moves,
        decisions,
    })
}

/// Read every complete game from a compact file.
///
/// Like the shard reader, a half-written trailing record stops the read rather
/// than failing it, so an interrupted run still yields everything it finished.
pub fn read_games(path: &str) -> Result<Vec<GameRecord>, CompactError> {
    let bytes = std::fs::read(path).map_err(|e| CompactError::Malformed(e.to_string()))?;
    let magic = bytes
        .get(0..4)
        .and_then(|c| c.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or(CompactError::Truncated)?;
    if magic != MAGIC {
        return Err(CompactError::NotACompactFile);
    }
    let mut c = Cursor { b: &bytes, p: 4 };
    let mut out = Vec::new();
    while c.p < bytes.len() {
        let start = c.p;
        match decode_game(&mut c) {
            Some(g) => out.push(g),
            None => {
                c.p = start;
                break;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------- expanding

/// Replay a game and rebuild the training examples it stands for.
///
/// Every recorded ply must land on a real decision during replay. If it does
/// not, the data and the engine disagree — most likely the rules changed since
/// the data was generated — and this fails loudly rather than emitting features
/// encoded from the wrong position, which would poison training invisibly.
pub fn expand_game(rec: &GameRecord, index: usize) -> Result<Vec<Example>, CompactError> {
    let mut game = Game::new(&rec.kingdom, rec.players as usize, rec.seed).map_err(|e| {
        CompactError::ReplayDiverged {
            game: index,
            detail: e.to_string(),
        }
    })?;

    let mut out = Vec::with_capacity(rec.decisions.len());
    let mut wanted = rec.decisions.iter().peekable();

    for (ply, mv) in rec.moves.iter().enumerate() {
        let Some(d) = game.decision().cloned() else {
            return Err(CompactError::ReplayDiverged {
                game: index,
                detail: format!("game ended at ply {ply}, {} moves remained", rec.moves.len() - ply),
            });
        };

        if let Some(rd) = wanted.peek() {
            if rd.ply as usize == ply {
                let rd = wanted.next().unwrap();
                out.push(Example {
                    features: features::encode(&game.state, d.player, &d),
                    policy: rd.policy.clone(),
                    outcome: rd.outcome,
                    td_target: rd.td_target,
                });
            }
        }

        game.apply(*mv).map_err(|e| CompactError::ReplayDiverged {
            game: index,
            detail: format!("move {mv} at ply {ply} was rejected: {e}"),
        })?;
    }

    if wanted.next().is_some() {
        return Err(CompactError::ReplayDiverged {
            game: index,
            detail: "recorded decisions past the end of the move list".into(),
        });
    }
    Ok(out)
}

/// Read a compact file and expand every game in it into training examples.
pub fn read_examples(path: &str) -> Result<Vec<Example>, CompactError> {
    let games = read_games(path)?;
    let mut out = Vec::new();
    for (i, g) in games.iter().enumerate() {
        out.extend(expand_game(g, i)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::HeuristicEvaluator;
    use crate::selfplay::{play_selfplay_game_recorded, play_selfplay_game_with_lambda};
    use crate::MctsConfig;
    use dominion_core::Rng;

    fn cfg() -> MctsConfig {
        MctsConfig {
            worlds: 1,
            iterations: 20,
            ..Default::default()
        }
    }

    /// The whole format rests on this: expanding a compact record must
    /// reproduce what generating examples directly would have produced. If the
    /// replayed features drift from the originals, training silently learns
    /// from positions that never occurred.
    #[test]
    fn expanding_a_record_reproduces_direct_generation() {
        let kingdom = dominion_core::Game::random_kingdom(&mut Rng::new(3));

        let mut rng = Rng::new(88);
        let direct =
            play_selfplay_game_with_lambda(&kingdom, 2, 5, &cfg(), &HeuristicEvaluator, &mut rng, 0.9);

        let mut rng = Rng::new(88);
        let record =
            play_selfplay_game_recorded(&kingdom, 2, 5, &cfg(), &HeuristicEvaluator, &mut rng, 0.9);

        let expanded = expand_game(&record, 0).expect("replays cleanly");

        assert_eq!(expanded.len(), direct.len(), "different number of examples");
        assert!(!expanded.is_empty());

        for (i, (a, b)) in expanded.iter().zip(&direct).enumerate() {
            assert_eq!(a.features, b.features, "features differ at example {i}");
            assert_eq!(a.policy.len(), b.policy.len(), "policy length differs at {i}");
            for ((m1, p1), (m2, p2)) in a.policy.iter().zip(&b.policy) {
                assert_eq!(m1, m2, "policy move differs at {i}");
                // Probabilities go through u16 fixed point on the way out.
                assert!((p1 - p2).abs() < 1e-4, "policy prob differs at {i}");
            }
            assert!((a.outcome - b.outcome).abs() < 1e-2);
            assert!((a.td_target - b.td_target).abs() < 1e-4);
        }
    }

    #[test]
    fn records_round_trip_through_a_file() {
        let kingdom = dominion_core::Game::random_kingdom(&mut Rng::new(1));
        let mut rng = Rng::new(4);
        let games: Vec<GameRecord> = (0..3)
            .map(|i| {
                play_selfplay_game_recorded(
                    &kingdom, 2, i, &cfg(), &HeuristicEvaluator, &mut rng, 0.9,
                )
            })
            .collect();

        let path = std::env::temp_dir()
            .join(format!("dominion-compact-{}.bin", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let _ = std::fs::remove_file(&path);

        // Written in two batches, the way a live run flushes.
        append_games(&path, &games[..2]).unwrap();
        append_games(&path, &games[2..]).unwrap();

        let back = read_games(&path).unwrap();
        assert_eq!(back.len(), 3);
        for (a, b) in back.iter().zip(&games) {
            assert_eq!(a.kingdom, b.kingdom);
            assert_eq!(a.seed, b.seed);
            assert_eq!(a.moves, b.moves);
            assert_eq!(a.decisions.len(), b.decisions.len());
        }

        std::fs::remove_file(&path).unwrap();
    }

    /// The size claim that justifies the whole format.
    #[test]
    fn a_game_is_far_smaller_than_its_examples() {
        let kingdom = dominion_core::Game::random_kingdom(&mut Rng::new(2));
        let mut rng = Rng::new(7);
        let record =
            play_selfplay_game_recorded(&kingdom, 2, 9, &cfg(), &HeuristicEvaluator, &mut rng, 0.9);

        let mut encoded = Vec::new();
        encode_game(&record, &mut encoded);

        let examples = expand_game(&record, 0).unwrap();
        let shard_bytes = examples.len() * (crate::features::FEATURE_DIM * 4 + 44);

        let ratio = shard_bytes as f64 / encoded.len() as f64;
        assert!(
            ratio > 10.0,
            "expected at least a 10x saving, got {ratio:.1}x \
             ({} bytes vs {} bytes of examples)",
            encoded.len(),
            shard_bytes
        );
    }

    /// A truncated tail costs the last game, not the file.
    #[test]
    fn a_truncated_file_keeps_its_complete_games() {
        let kingdom = dominion_core::Game::random_kingdom(&mut Rng::new(5));
        let mut rng = Rng::new(6);
        let games: Vec<GameRecord> = (0..3)
            .map(|i| {
                play_selfplay_game_recorded(
                    &kingdom, 2, i, &cfg(), &HeuristicEvaluator, &mut rng, 0.9,
                )
            })
            .collect();

        let path = std::env::temp_dir()
            .join(format!("dominion-compact-trunc-{}.bin", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let _ = std::fs::remove_file(&path);
        append_games(&path, &games).unwrap();

        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() - 50]).unwrap();
        let back = read_games(&path).unwrap();
        assert_eq!(back.len(), 2, "should keep the two complete games");

        std::fs::remove_file(&path).unwrap();
    }

    /// Data that does not replay must fail loudly, never yield wrong features.
    #[test]
    fn a_record_that_cannot_replay_is_rejected() {
        let kingdom = dominion_core::Game::random_kingdom(&mut Rng::new(8));
        let mut rng = Rng::new(2);
        let mut record =
            play_selfplay_game_recorded(&kingdom, 2, 12, &cfg(), &HeuristicEvaluator, &mut rng, 0.9);

        // Corrupt a move so replay hits an illegal position.
        record.moves[4] = Move::Buy(Card::Province);
        let err = expand_game(&record, 0);
        assert!(matches!(err, Err(CompactError::ReplayDiverged { .. })), "{err:?}");

        // And a decision pointing past the end must be caught too.
        let mut record2 =
            play_selfplay_game_recorded(&kingdom, 2, 13, &cfg(), &HeuristicEvaluator, &mut rng, 0.9);
        record2.decisions.push(RecordedDecision {
            ply: 60_000,
            policy: vec![(Move::Done, 1.0)],
            outcome: 1.0,
            td_target: 1.0,
        });
        assert!(matches!(
            expand_game(&record2, 0),
            Err(CompactError::ReplayDiverged { .. })
        ));
    }

    #[test]
    fn garbage_is_rejected() {
        let path = std::env::temp_dir()
            .join(format!("dominion-compact-junk-{}.bin", std::process::id()))
            .to_string_lossy()
            .into_owned();
        std::fs::write(&path, [1u8, 2, 3, 4, 5, 6]).unwrap();
        assert!(matches!(read_games(&path), Err(CompactError::NotACompactFile)));
        std::fs::remove_file(&path).unwrap();
    }
}
