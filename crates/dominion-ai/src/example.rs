//! Training examples and the shard file they are exchanged in.
//!
//! Self-play data is meant to be generated on more than one machine at once —
//! that is the whole point of running it on both a laptop and here — so each
//! producer writes its own shard under a name nobody else will pick, and
//! training reads however many shards happen to exist. No merge conflicts,
//! no coordination beyond picking distinct file names.

use dominion_core::Move;

use crate::features::FEATURE_DIM;

/// One decision from a finished self-play game: the position, the search's
/// verdict on it, and two different value targets for it.
#[derive(Clone, Debug)]
pub struct Example {
    pub features: [f32; FEATURE_DIM],
    /// The moves the search considered at this decision (post-restriction),
    /// paired with the fraction of visits each received. Sums to 1.
    pub policy: Vec<(Move, f32)>,
    /// The eventual result for the player who was deciding, in `[0, 1]`.
    /// The pure Monte Carlo target: unbiased, but a single bit of signal
    /// shared across every one of a game's ~240 decisions.
    pub outcome: f32,
    /// TD(lambda) target, mixing the outcome with the search's own value
    /// estimates further along the trajectory. Both are stored so the two can
    /// be compared without regenerating self-play data.
    pub td_target: f32,
}

impl Example {
    /// The target to train the value head against, for a given TD/MC choice.
    pub fn value_target(&self, use_td: bool) -> f32 {
        if use_td {
            self.td_target
        } else {
            self.outcome
        }
    }
}

/// v1 stored a single `value` field. v2 stores `outcome` and `td_target`
/// separately; v1 files stay readable, with both fields set to the old value.
const MAGIC_V1: u32 = 0xD0A1_5EAF;
const MAGIC_V2: u32 = 0xD0A1_5EB2;
const MAGIC: u32 = MAGIC_V2;

/// Append examples to a shard file, creating it if it does not exist.
pub fn append_shard(path: &str, examples: &[Example]) -> std::io::Result<()> {
    use std::io::Write;
    let is_new = !std::path::Path::new(path).exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if is_new {
        f.write_all(&MAGIC.to_le_bytes())?;
    }
    for ex in examples {
        let mut rec = Vec::new();
        for &v in &ex.features {
            rec.extend_from_slice(&v.to_le_bytes());
        }
        rec.extend_from_slice(&(ex.policy.len() as u32).to_le_bytes());
        for &(mv, p) in &ex.policy {
            rec.extend_from_slice(&(mv.index() as u32).to_le_bytes());
            rec.extend_from_slice(&p.to_le_bytes());
        }
        rec.extend_from_slice(&ex.outcome.to_le_bytes());
        rec.extend_from_slice(&ex.td_target.to_le_bytes());
        f.write_all(&rec)?;
    }
    Ok(())
}

/// Read every example out of one shard file.
pub fn read_shard(path: &str) -> std::io::Result<Vec<Example>> {
    let bytes = std::fs::read(path)?;
    parse_shard(&bytes).ok_or_else(|| {
        std::io::Error::other(format!("{path}: not a valid shard file"))
    })
}

/// Read and concatenate every shard matching a glob-free list of paths (the
/// caller expands wildcards; this just reads what it is given).
pub fn read_shards(paths: &[String]) -> std::io::Result<Vec<Example>> {
    let mut all = Vec::new();
    for p in paths {
        all.extend(read_shard(p)?);
    }
    Ok(all)
}

fn parse_shard(bytes: &[u8]) -> Option<Vec<Example>> {
    let mut pos = 0usize;
    let magic = u32::from_le_bytes(bytes.get(0..4)?.try_into().ok()?);
    let v2 = match magic {
        MAGIC_V2 => true,
        MAGIC_V1 => false,
        _ => return None,
    };
    pos += 4;

    let read_f32 = |bytes: &[u8], pos: &mut usize| -> Option<f32> {
        let c = bytes.get(*pos..*pos + 4)?;
        *pos += 4;
        Some(f32::from_le_bytes(c.try_into().ok()?))
    };
    let read_u32 = |bytes: &[u8], pos: &mut usize| -> Option<u32> {
        let c = bytes.get(*pos..*pos + 4)?;
        *pos += 4;
        Some(u32::from_le_bytes(c.try_into().ok()?))
    };

    let mut out = Vec::new();
    while pos < bytes.len() {
        let mut features = [0.0f32; FEATURE_DIM];
        for f in &mut features {
            *f = read_f32(bytes, &mut pos)?;
        }
        let n = read_u32(bytes, &mut pos)? as usize;
        let mut policy = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = read_u32(bytes, &mut pos)? as usize;
            let p = read_f32(bytes, &mut pos)?;
            let mv = Move::from_index(idx)?;
            policy.push((mv, p));
        }
        let outcome = read_f32(bytes, &mut pos)?;
        // v1 knew only the final outcome, so it is the best TD target it can
        // offer; training on such a shard is simply pure Monte Carlo.
        let td_target = if v2 { read_f32(bytes, &mut pos)? } else { outcome };
        out.push(Example {
            features,
            policy,
            outcome,
            td_target,
        });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dominion_core::Card;

    fn sample(value: f32) -> Example {
        let mut features = [0.0f32; FEATURE_DIM];
        features[3] = 0.5;
        Example {
            features,
            policy: vec![(Move::Buy(Card::Gold), 0.7), (Move::Buy(Card::Silver), 0.3)],
            outcome: value,
            td_target: value * 0.5 + 0.25,
        }
    }

    #[test]
    fn a_shard_round_trips_through_a_file() {
        let dir = std::env::temp_dir();
        let path = dir
            .join(format!("dominion-shard-test-{}.bin", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let _ = std::fs::remove_file(&path);

        let batch1 = vec![sample(0.0), sample(1.0)];
        let batch2 = vec![sample(0.5)];
        append_shard(&path, &batch1).unwrap();
        append_shard(&path, &batch2).unwrap();

        let all = read_shard(&path).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].outcome, 0.0);
        assert_eq!(all[0].td_target, 0.25);
        assert_eq!(all[2].outcome, 0.5);
        assert_eq!(all[0].policy, batch1[0].policy);
        assert_eq!(all[0].features, batch1[0].features);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn garbage_is_rejected_not_panicked_on() {
        assert!(parse_shard(&[9, 9, 9]).is_none());
        assert!(parse_shard(&[]).is_none());
    }

    /// The 1500-game shard generated before TD targets existed must stay
    /// usable; a v1 record simply has no bootstrapped target, so both fields
    /// carry the final outcome.
    #[test]
    fn v1_shards_are_still_readable() {
        let mut bytes = MAGIC_V1.to_le_bytes().to_vec();
        for v in [0.0f32; FEATURE_DIM] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(Move::Buy(Card::Gold).index() as u32).to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&0.75f32.to_le_bytes());

        let parsed = parse_shard(&bytes).expect("v1 shard parses");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].outcome, 0.75);
        assert_eq!(parsed[0].td_target, 0.75);
        assert_eq!(parsed[0].value_target(true), 0.75);
        assert_eq!(parsed[0].value_target(false), 0.75);
    }
}
