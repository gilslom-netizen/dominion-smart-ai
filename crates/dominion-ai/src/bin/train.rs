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
use dominion_ai::{compact, example, MctsConfig, Net};
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

const USAGE: &str = "\
train the policy/value network on self-play data

usage: train [options]

  --data <dir>       corpus to read (default selfplay-data)
  --net-in <path>    start from this checkpoint instead of random init
  --net-out <path>   where to save (default models/net.bin)
  --epochs <n>       passes over the data (default 6)
  --lr <f>           learning rate (default 0.01)
  --limit <n>        cap examples, for step-matched comparisons
  --eval-games <n>   games to measure against the heuristic; 0 skips
  --mc               train the value head on final outcomes, not TD targets";

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        std::process::exit(0);
    }
    const KNOWN: &[&str] = &[
        "--data", "--net-in", "--net-out", "--epochs", "--lr", "--limit",
        "--eval-games", "--mc",
    ];
    for a in raw.iter().filter(|a| a.starts_with("--")) {
        if !KNOWN.contains(&a.as_str()) {
            eprintln!("unknown option {a}\n\n{USAGE}");
            std::process::exit(2);
        }
    }

    let net_in = parse_flag("--net-in");
    let net_out = parse_flag("--net-out").unwrap_or_else(|| "models/net.bin".into());
    let epochs: u32 = parse_flag("--epochs").and_then(|s| s.parse().ok()).unwrap_or(6);
    let lr: f32 = parse_flag("--lr").and_then(|s| s.parse().ok()).unwrap_or(0.01);
    let eval_games: u32 = parse_flag("--eval-games").and_then(|s| s.parse().ok()).unwrap_or(60);
    // TD targets are the default; --mc trains the value head on the raw final
    // outcome instead, so the two can be compared on identical data.
    let use_td = !std::env::args().any(|a| a == "--mc");
    // Which directory to read. Controlled experiments need to train on one
    // specific corpus rather than on everything that happens to be lying
    // around — the difference between a comparison and a coincidence.
    let data_dir = parse_flag("--data").unwrap_or_else(|| "selfplay-data".into());
    // Cap on examples used, so two corpora of different sizes can be compared
    // at a matched number of gradient steps rather than matched epochs.
    let limit: Option<usize> = parse_flag("--limit").and_then(|s| s.parse().ok());

    let mut shard_paths: Vec<String> = Vec::new();
    let mut gamelog_paths: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&data_dir) {
        for entry in rd.filter_map(|e| e.ok()) {
            let path = entry.path();
            let name = path.to_string_lossy().into_owned();
            match path.extension().and_then(|x| x.to_str()) {
                Some("shard") => shard_paths.push(name),
                Some("gamelog") => gamelog_paths.push(name),
                _ => {}
            }
        }
    }
    shard_paths.sort();
    gamelog_paths.sort();

    if shard_paths.is_empty() && gamelog_paths.is_empty() {
        eprintln!("nothing in {data_dir}/ — run bin/selfplay first");
        std::process::exit(1);
    }

    let mut examples = Vec::new();
    // Compact game logs, expanded by replaying each game. These are the ones
    // that travel between machines.
    for p in &gamelog_paths {
        let got = compact::read_examples(p).unwrap_or_else(|e| {
            eprintln!("failed to read {p}: {e}");
            std::process::exit(1);
        });
        println!("  {p}: {} examples (replayed)", got.len());
        examples.extend(got);
    }
    // Legacy expanded shards, kept readable so earlier data is not stranded.
    for p in &shard_paths {
        let got = example::read_shard(p).unwrap_or_else(|e| {
            eprintln!("failed to read {p}: {e}");
            std::process::exit(1);
        });
        println!("  {p}: {} examples", got.len());
        examples.extend(got);
    }
    if let Some(n) = limit {
        if examples.len() > n {
            // Shuffle before truncating: taking a prefix would take whole
            // games in generation order, not a sample of positions.
            let mut r = Rng::new(0xC0FFEE);
            r.shuffle(&mut examples);
            examples.truncate(n);
            println!("limited to {n} examples");
        }
    }
    // Single-option positions train the value head but not the policy head
    // (see Net::train_step). They are kept: dropping them cost 61.7% of the
    // value head's data and measurably hurt it.
    let forced = examples.iter().filter(|e| e.policy.len() <= 1).count();
    println!("{} training examples", examples.len());
    if forced > 0 {
        println!(
            "  {forced} ({:.1}%) are forced positions: value head only",
            100.0 * forced as f64 / examples.len().max(1) as f64
        );
    }
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
        // Policy loss is averaged over the examples that actually train it,
        // so it stays comparable no matter how many forced positions the
        // corpus happens to contain.
        let n = examples.len().max(1) as f64;
        let n_policy = (examples.len() - forced).max(1) as f64;
        println!(
            "epoch {epoch}: policy loss {:.4}, value loss {:.4}",
            policy_loss / n_policy,
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
    // Skipped at --eval-games 0, which is how a controlled experiment asks for
    // a trained network without paying for a measurement it will not use.
    if eval_games == 0 {
        return;
    }
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
