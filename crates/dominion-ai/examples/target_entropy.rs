//! Is the policy head actually underfitting, or has it already learned
//! everything its targets contain?
//!
//! Cross-entropy against a soft target cannot go below the entropy of that
//! target. Policy loss sat at ~0.34 across every configuration tried — small
//! network, 7x bigger network, quarter of the data, all of it — which is the
//! signature of a floor rather than of underfitting. This measures where that
//! floor is.
//!
//! If the mean target entropy is ~0.34, the network is already reproducing the
//! search's policy about as exactly as the target permits, and the way to a
//! better policy is better *targets* — a deeper search when generating data —
//! not a bigger network or more examples.

use dominion_ai::{compact, example, Example};

fn load_all() -> Vec<Example> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir("selfplay-data") else {
        return out;
    };
    let mut paths: Vec<std::path::PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    for p in paths {
        let name = p.to_string_lossy().into_owned();
        match p.extension().and_then(|x| x.to_str()) {
            Some("gamelog") => out.extend(compact::read_examples(&name).unwrap_or_default()),
            Some("shard") => out.extend(example::read_shard(&name).unwrap_or_default()),
            _ => {}
        }
    }
    out
}

fn entropy(policy: &[(dominion_core::Move, f32)]) -> f32 {
    -policy
        .iter()
        .map(|(_, p)| if *p > 0.0 { p * p.ln() } else { 0.0 })
        .sum::<f32>()
}

fn main() {
    let examples = load_all();
    if examples.is_empty() {
        eprintln!("nothing in selfplay-data/");
        std::process::exit(1);
    }

    let mut total = 0.0f64;
    let mut by_options: Vec<(usize, f64, usize)> = Vec::new(); // (n_options, sum H, count)
    let mut peaked = 0usize;

    for ex in &examples {
        let h = entropy(&ex.policy);
        total += h as f64;
        let n = ex.policy.len();
        // A target where one move holds most of the mass leaves the network
        // very little to learn beyond "pick that one".
        if ex.policy.iter().any(|(_, p)| *p > 0.9) {
            peaked += 1;
        }
        match by_options.iter_mut().find(|(k, _, _)| *k == n) {
            Some(slot) => {
                slot.1 += h as f64;
                slot.2 += 1;
            }
            None => by_options.push((n, h as f64, 1)),
        }
    }

    let mean = total / examples.len() as f64;
    println!("{} examples", examples.len());
    println!("mean target entropy: {mean:.4}");
    println!(
        "targets with one move above 90% of the visits: {:.1}%",
        100.0 * peaked as f64 / examples.len() as f64
    );

    by_options.sort_by_key(|(n, _, _)| *n);
    println!("\n  options   mean entropy   share of data");
    for (n, sum, count) in by_options.iter().take(14) {
        println!(
            "  {n:>7}   {:>12.4}   {:>12.1}%",
            sum / *count as f64,
            100.0 * *count as f64 / examples.len() as f64
        );
    }

    println!("\nPolicy cross-entropy cannot fall below this mean. Compare it against");
    println!("the training loss: if they match, the policy head has converged and");
    println!("the limit is the quality of the search that produced the targets.");
}
