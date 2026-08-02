//! Play two trained networks directly against each other.
//!
//! ```text
//! cargo run --release --example net_vs_net -- a.bin b.bin [pairs]
//! ```
//!
//! Measuring each network against a common third party (the heuristic) and
//! comparing the two win rates is a much weaker test than it looks: the two
//! results carry independent noise, so the error on their *difference* is the
//! two errors added in quadrature. Playing the networks against each other
//! removes that entirely — every game contributes directly to the comparison
//! being made, and the harness pairs seeds and swaps seats on top.

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsConfig, Net};
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::Agent;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: net_vs_net <a.bin> <b.bin> [pairs]");
        std::process::exit(1);
    }
    let pairs: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);

    let a = Net::load(&args[0]).expect("load first network");
    let b = Net::load(&args[1]).expect("load second network");
    let cfg = MctsConfig {
        worlds: 4,
        iterations: 200,
        ..Default::default()
    };
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let (a_ref, b_ref) = (&a, &b);
    let res = run_match_parallel(
        move || Box::new(NetMctsAgent::new(cfg, a_ref)) as Box<dyn Agent>,
        move || Box::new(NetMctsAgent::new(cfg, b_ref)) as Box<dyn Agent>,
        pairs,
        0x9A11,
        &Kingdoms::Random,
        cores,
    );

    println!("{} vs {}", args[0], args[1]);
    println!("{res}");
    let edge = (res.win_rate_a() - 0.5).abs();
    println!(
        "{:.1} standard errors from even",
        if res.stderr() > 0.0 { edge / res.stderr() } else { 0.0 }
    );
}
