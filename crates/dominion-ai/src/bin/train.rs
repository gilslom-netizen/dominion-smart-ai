//! Train the network on whatever self-play shards exist, and report how it
//! did against the heuristic.
//!
//! ```text
//! cargo run --release --bin train -- --net-in models/net.bin --net-out models/net.bin
//! ```
//!
//! Reads every `selfplay-data/*.shard`, regardless of which machine or
//! `--tag` produced it — that is the entire point of the shard format: this
//! step is where two machines' self-play work actually combines into one
//! stronger network.

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{example, MctsConfig, Net};
use dominion_bots::match_runner::run_match_parallel;
use dominion_bots::policy::HeuristicBot;
use dominion_bots::{Agent, Kingdoms};
use dominion_core::Rng;

fn parse_flag(flag: &str) -> Option<String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    raw.iter()
        .position(|a| a == flag)
        .and_then(|i| raw.get(i + 1))
        .cloned()
}

fn main() {
    let net_in = parse_flag("--net-in");
    let net_out = parse_flag("--net-out").unwrap_or_else(|| "models/net.bin".into());
    let epochs: u32 = parse_flag("--epochs").and_then(|s| s.parse().ok()).unwrap_or(6);
    let lr: f32 = parse_flag("--lr").and_then(|s| s.parse().ok()).unwrap_or(0.01);
    let eval_games: u32 = parse_flag("--eval-games").and_then(|s| s.parse().ok()).unwrap_or(60);
    // TD targets are the default; --mc trains the value head on the raw final
    // outcome instead, so the two can be compared on identical data.
    let use_td = !std::env::args().any(|a| a == "--mc");

    let shard_paths: Vec<String> = std::fs::read_dir("selfplay-data")
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x == "shard").unwrap_or(false))
                .map(|p| p.to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();

    if shard_paths.is_empty() {
        eprintln!("no shards found in selfplay-data/ — run bin/selfplay first");
        std::process::exit(1);
    }

    println!("reading {} shard(s):", shard_paths.len());
    for p in &shard_paths {
        println!("  {p}");
    }
    let mut examples = example::read_shards(&shard_paths).unwrap_or_else(|e| {
        eprintln!("failed to read shards: {e}");
        std::process::exit(1);
    });
    println!("{} training examples", examples.len());
    println!(
        "value target: {}",
        if use_td { "TD(lambda)" } else { "Monte Carlo (final outcome)" }
    );
    // If every TD target equals its outcome, the shards predate TD targets and
    // the two modes would be identical — worth saying rather than silently
    // reporting a pointless comparison.
    if use_td && examples.iter().all(|e| e.td_target == e.outcome) {
        println!("  (these shards are v1: no bootstrapped targets, so this is Monte Carlo)");
    }

    let mut rng = Rng::new(0x7EA1);
    let mut net = match &net_in {
        Some(p) => Net::load(p).unwrap_or_else(|e| {
            eprintln!("cannot load {p}: {e}");
            std::process::exit(1);
        }),
        None => {
            println!("no --net-in given, starting from a freshly initialized network");
            Net::new(&mut rng)
        }
    };

    for epoch in 0..epochs {
        rng.shuffle(&mut examples);
        let mut policy_loss = 0.0f64;
        let mut value_loss = 0.0f64;
        for ex in &examples {
            let indices: Vec<usize> = ex.policy.iter().map(|(mv, _)| mv.index()).collect();
            let targets: Vec<f32> = ex.policy.iter().map(|(_, p)| *p).collect();
            let target = ex.value_target(use_td);
            let (pl, vl) = net.train_step(&ex.features, &indices, &targets, target, lr);
            policy_loss += pl as f64;
            value_loss += vl as f64;
        }
        let n = examples.len().max(1) as f64;
        println!(
            "epoch {epoch}: policy loss {:.4}, value loss {:.4}",
            policy_loss / n,
            value_loss / n
        );
    }

    std::fs::create_dir_all(
        std::path::Path::new(&net_out)
            .parent()
            .unwrap_or_else(|| std::path::Path::new(".")),
    )
    .ok();
    net.save(&net_out).unwrap_or_else(|e| {
        eprintln!("failed to save {net_out}: {e}");
        std::process::exit(1);
    });
    println!("saved trained network to {net_out}");

    // How good is it? Cheap, fast diagnostic: the network's own raw priors and
    // value used to drive a light search, measured against the heuristic.
    println!("\nevaluating: NetMCTS vs Heuristic ({} games)...", eval_games * 2);
    let cfg = MctsConfig {
        worlds: 4,
        iterations: 200,
        ..Default::default()
    };
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let net_ref = &net;
    let res = run_match_parallel(
        move || Box::new(NetMctsAgent::new(cfg, net_ref)) as Box<dyn Agent>,
        || Box::new(HeuristicBot) as Box<dyn Agent>,
        eval_games,
        0xF00D,
        &Kingdoms::Random,
        cores,
    );
    println!("{res}");
}
