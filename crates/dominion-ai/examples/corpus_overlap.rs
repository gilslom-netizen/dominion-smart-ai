//! Do these corpora contain the same games?
//!
//! Worth checking before pooling self-play from several machines, because the
//! failure is silent. When `--seed` defaulted to 0, two agents running the same
//! command produced *identical* games — and the files still differed byte for
//! byte, because threads finish in a different order each run. Comparing sizes
//! or checksums would have shown two healthy-looking distinct files.
//!
//! A game is identified by its kingdom and seed, which together determine it
//! completely, so that is what gets compared.
//!
//! ```text
//! cargo run --release --example corpus_overlap -- selfplay-data/*.gamelog
//! ```

use std::collections::HashSet;

use dominion_ai::compact;

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: corpus_overlap <file.gamelog>...");
        std::process::exit(1);
    }

    let mut corpora: Vec<(String, HashSet<(u64, Vec<u8>)>)> = Vec::new();
    for path in &paths {
        let games = match compact::read_games(path) {
            Ok(g) => g,
            Err(e) => {
                println!("{path}: unreadable — {e}");
                continue;
            }
        };
        let ids: HashSet<(u64, Vec<u8>)> = games
            .iter()
            .map(|g| (g.seed, g.kingdom.iter().map(|c| *c as u8).collect()))
            .collect();
        let name = path.rsplit('/').next().unwrap_or(path).to_string();
        println!("{name}: {} games ({} distinct)", games.len(), ids.len());
        if ids.len() < games.len() {
            println!(
                "  note: {} repeats within this file",
                games.len() - ids.len()
            );
        }
        corpora.push((name, ids));
    }

    if corpora.len() < 2 {
        return;
    }

    println!("\npairwise overlap:");
    let mut worst = 0.0f64;
    for i in 0..corpora.len() {
        for j in i + 1..corpora.len() {
            let shared = corpora[i].1.intersection(&corpora[j].1).count();
            let smaller = corpora[i].1.len().min(corpora[j].1.len()).max(1);
            let pct = 100.0 * shared as f64 / smaller as f64;
            worst = worst.max(pct);
            println!(
                "  {:<32} {:<32} {shared:>6} shared  ({pct:.0}% of the smaller)",
                corpora[i].0, corpora[j].0
            );
        }
    }

    let union: HashSet<_> = corpora.iter().flat_map(|(_, s)| s.iter()).collect();
    let total: usize = corpora.iter().map(|(_, s)| s.len()).sum();
    println!(
        "\n{total} games across all files, {} distinct — {:.0}% of the effort is duplicated",
        union.len(),
        100.0 * (total - union.len()) as f64 / total.max(1) as f64
    );
    if worst > 50.0 {
        println!(
            "\nHigh overlap. Check that each machine used a distinct --tag: the seed\n\
             is derived from it, and two runs sharing a tag replay the same games."
        );
    }
}
