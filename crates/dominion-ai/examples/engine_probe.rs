//! Is the engine actually better, or is money right?
//!
//! The search prefers money everywhere, and two very different things would
//! explain that. Either a thin-deck engine is stronger and the search cannot
//! see it — Chapel's payoff arrives five to eight turns after its cost, which
//! is past the tree and therefore priced by a rollout that keeps buying money
//! — or money genuinely is the stronger line in Base 2E, as it is widely held
//! to be, and preferring it is correct.
//!
//! Assuming the first without checking is how a project spends a week teaching
//! its AI to play worse. So this measures the engine directly:
//!
//! 1. the engine menu against every menu on the ladder, which says whether it
//!    is strong at all; and
//! 2. the network against the engine menu, which says whether the AI can beat
//!    the thing it refuses to build.
//!
//! A fixed priority list pilots an engine badly, so a weak showing is a lower
//! bound on the engine's strength rather than a verdict on engines.

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsConfig, Net};
use dominion_bots::buy::{chapel_engine, ladder, required_kingdom, MenuBot};
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::Agent;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "models/net.bin".into());
    let pairs: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("== the engine against the ladder ==");
    println!("{:<18} {:>9} {:>10}", "opponent", "win rate", "±");
    let mut total = 0.0f64;
    let mut n = 0u32;
    for menu in ladder() {
        let name = menu.name.clone();
        let mut must = required_kingdom(&chapel_engine());
        must.extend(required_kingdom(&menu));
        let res = run_match_parallel(
            || Box::new(MenuBot::new(chapel_engine())) as Box<dyn Agent>,
            move || Box::new(MenuBot::new(menu.clone())) as Box<dyn Agent>,
            pairs,
            0xBEEF,
            &Kingdoms::RandomWith(must),
            cores,
        );
        println!(
            "{:<18} {:>8.2}% {:>9.2}",
            name,
            100.0 * res.win_rate_a(),
            100.0 * res.stderr()
        );
        total += res.win_rate_a() as f64;
        n += 1;
    }
    println!(
        "\naverage: {:.2}%  (the hand-written heuristic averages 64.1% here)\n",
        100.0 * total / n.max(1) as f64
    );

    println!("== the network against the engine ==");
    let net = Net::load(&path).expect("load network");
    let cfg = MctsConfig {
        worlds: 4,
        iterations: 200,
        ..Default::default()
    };
    let net_ref = &net;
    let res = run_match_parallel(
        move || Box::new(NetMctsAgent::new(cfg, net_ref)) as Box<dyn Agent>,
        || Box::new(MenuBot::new(chapel_engine())) as Box<dyn Agent>,
        pairs,
        0xC0FFEE,
        &Kingdoms::RandomWith(required_kingdom(&chapel_engine())),
        cores,
    );
    let sigma = (res.win_rate_a() - 0.5).abs() / res.stderr().max(1e-9);
    println!("  {res}");
    println!("  {sigma:.1} standard errors from even");
}
