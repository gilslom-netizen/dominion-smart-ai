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
/// verdict on it, and the eventual outcome.
#[derive(Clone, Debug)]
pub struct Example {
    pub features: [f32; FEATURE_DIM],
    /// The moves the search considered at this decision (post-restriction),
    /// paired with the fraction of visits each received. Sums to 1.
    pub policy: Vec<(Move, f32)>,
    /// The eventual result for the player who was deciding, in `[0, 1]`.
    pub value: f32,
}

const MAGIC: u32 = 0xD0A1_5EAF;

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
        rec.extend_from_slice(&ex.value.to_le_bytes());
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
    if magic != MAGIC {
        return None;
    }
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
        let value = read_f32(bytes, &mut pos)?;
        out.push(Example {
            features,
            policy,
            value,
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
            value,
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
        assert_eq!(all[0].value, 0.0);
        assert_eq!(all[2].value, 0.5);
        assert_eq!(all[0].policy, batch1[0].policy);
        assert_eq!(all[0].features, batch1[0].features);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn garbage_is_rejected_not_panicked_on() {
        assert!(parse_shard(&[9, 9, 9]).is_none());
        assert!(parse_shard(&[]).is_none());
    }
}
