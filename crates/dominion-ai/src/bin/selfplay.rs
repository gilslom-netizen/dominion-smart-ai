//! Generate a shard of self-play training data.
//!
//! Meant to run on more than one machine at once: give each machine a distinct
//! `--tag` and they will never write the same file, so there is nothing to
//! coordinate before `train` reads everything back.
//!
//! ```text
//! cargo run --release --bin selfplay -- --games 500 --threads 12 --tag laptop
//! cargo run --release --bin selfplay -- --games 200 --threads 4  --tag cloud --net models/net.bin
//! ```
//!
//! Without `--net`, generation uses the heuristic-guided search already
//! measured in the README (60.4% vs the heuristic). With `--net`, it uses that
//! network's policy and value instead — the loop that is supposed to make each
//! generation stronger than the last.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use dominion_ai::evaluator::{HeuristicEvaluator, NetEvaluator};
use dominion_ai::{example, play_selfplay_game, MctsConfig, Net};
use dominion_core::{Game, Rng};

struct Args {
    games: u32,
    threads: usize,
    tag: String,
    net_path: Option<String>,
    worlds: u32,
    iterations: u32,
    seed: u64,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| -> Option<String> {
        raw.iter()
            .position(|a| a == flag)
            .and_then(|i| raw.get(i + 1))
            .cloned()
    };
    Args {
        games: get("--games").and_then(|s| s.parse().ok()).unwrap_or(200),
        threads: get("--threads")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)),
        tag: get("--tag").unwrap_or_else(|| "default".into()),
        net_path: get("--net"),
        worlds: get("--worlds").and_then(|s| s.parse().ok()).unwrap_or(8),
        iterations: get("--iterations").and_then(|s| s.parse().ok()).unwrap_or(300),
        seed: get("--seed").and_then(|s| s.parse().ok()).unwrap_or(0),
    }
}

fn main() {
    let args = parse_args();
    std::fs::create_dir_all("selfplay-data").ok();
    let out_path = format!(
        "selfplay-data/{}-{}.shard",
        args.tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    let net = args.net_path.as_ref().map(|p| {
        Net::load(p).unwrap_or_else(|e| {
            eprintln!("cannot load {p}: {e}");
            std::process::exit(1);
        })
    });

    let cfg = MctsConfig {
        worlds: args.worlds,
        iterations: args.iterations,
        ..Default::default()
    };

    println!(
        "generating {} games on {} threads -> {out_path}{}",
        args.games,
        args.threads,
        if net.is_some() { " (net-guided)" } else { " (heuristic-guided)" }
    );

    let done = AtomicU32::new(0);
    let games = args.games;
    let start = Instant::now();

    let shards: Vec<Vec<dominion_ai::Example>> = std::thread::scope(|scope| {
        let net = &net;
        let done = &done;
        let handles: Vec<_> = (0..args.threads)
            .map(|t| {
                let lo = (games as usize * t / args.threads) as u32;
                let hi = (games as usize * (t + 1) / args.threads) as u32;
                scope.spawn(move || {
                    let mut rng = Rng::new(args.seed ^ (t as u64).wrapping_mul(0x9E3779B97F4A7C15));
                    let mut out = Vec::new();
                    for i in lo..hi {
                        let game_seed = args.seed
                            .wrapping_mul(0x2545F4914F6CDD1D)
                            .wrapping_add(i as u64);
                        let mut krng = Rng::new(game_seed ^ 0xABCD);
                        let kingdom = Game::random_kingdom(&mut krng);
                        let examples = match &net {
                            Some(n) => {
                                let eval = NetEvaluator { net: n };
                                play_selfplay_game(&kingdom, 2, game_seed, &cfg, &eval, &mut rng)
                            }
                            None => play_selfplay_game(
                                &kingdom,
                                2,
                                game_seed,
                                &cfg,
                                &HeuristicEvaluator,
                                &mut rng,
                            ),
                        };
                        out.extend(examples);
                        let n_done = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n_done % 10 == 0 || n_done == games {
                            let secs = start.elapsed().as_secs_f64();
                            eprintln!(
                                "{n_done}/{games} games  ({:.2} games/s)",
                                n_done as f64 / secs.max(1e-9)
                            );
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let all: Vec<dominion_ai::Example> = shards.into_iter().flatten().collect();
    example::append_shard(&out_path, &all).unwrap_or_else(|e| {
        eprintln!("failed to write {out_path}: {e}");
        std::process::exit(1);
    });

    println!(
        "wrote {} examples from {} games to {out_path} in {:.1}s",
        all.len(),
        args.games,
        start.elapsed().as_secs_f64()
    );
    println!("next: commit and push {out_path}, then run bin/train once shards from every machine are in.");
}
