//! Combine networks trained in parallel on different machines.
//!
//! ```text
//! cargo run --release --bin merge_nets -- models/net-alice.bin:3000 models/net-bob.bin:800
//! ```
//!
//! Each argument is a weights file, optionally followed by `:games` giving how
//! much self-play it was trained on. Contributions are weighted by that, so a
//! machine that generated 3000 games counts proportionally more than one that
//! managed 800. Without a `:games` suffix a file counts as 1.
//!
//! **This is only valid if every input was fine-tuned from the same starting
//! checkpoint.** That is the coordination rule the whole parallel-generation
//! scheme rests on: everyone pulls the current `models/net.bin`, trains from
//! it on their own fresh self-play, and pushes their result under their own
//! name. Averaging networks that started from different random initialisations
//! would destroy both, and nothing here can detect that — see
//! [`dominion_ai::Net::weighted_average`].

use dominion_ai::Net;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out_path = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "models/net.bin".into());

    let inputs: Vec<&String> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .filter(|a| *a != &out_path)
        .collect();

    if inputs.is_empty() {
        eprintln!("usage: merge_nets <net.bin[:games]>... [--out models/net.bin]");
        std::process::exit(1);
    }

    let mut nets = Vec::new();
    for spec in &inputs {
        let (path, weight) = match spec.rsplit_once(':') {
            Some((p, w)) => match w.parse::<f32>() {
                Ok(w) => (p, w),
                // A Windows path like C:\... has a colon that is not a weight.
                Err(_) => (spec.as_str(), 1.0),
            },
            None => (spec.as_str(), 1.0),
        };
        let net = Net::load(path).unwrap_or_else(|e| {
            eprintln!("cannot load {path}: {e}");
            std::process::exit(1);
        });
        println!("  {path}  weight {weight}");
        nets.push((net, weight));
    }

    let Some(merged) = Net::weighted_average(&nets) else {
        eprintln!("cannot merge: weights sum to zero, or architectures differ");
        std::process::exit(1);
    };

    std::fs::create_dir_all(
        std::path::Path::new(&out_path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new(".")),
    )
    .ok();
    merged.save(&out_path).unwrap_or_else(|e| {
        eprintln!("failed to write {out_path}: {e}");
        std::process::exit(1);
    });

    println!("merged {} networks into {out_path}", nets.len());
    println!("next: measure it before trusting it — a merge is not automatically an improvement.");
}
