//! Does unlocking the search from its prior actually make it play better?
//!
//! `unlock_sweep` established that prior temperature works as a mechanism:
//! at temp 4.0 / c 2.5 the top move's visit share falls from 72% to 41% and
//! 16x the budget changes the chosen move in 25% of decisions instead of 6%.
//! That is a diagnostic, not a result. Disagreeing with the prior is only
//! valuable if the search is *right* to disagree.
//!
//! Three questions, each with the same network on both sides so the only
//! variable is the search configuration:
//!
//! 1. At the unlocked setting, does 16x the budget beat 1x? This is the one
//!    that matters. Under the locked prior, 16x won 50.42% +/- 4.56 — the
//!    search could not convert compute into strength. If the unlock is real,
//!    this number moves.
//! 2. At equal budget, is the unlocked setting better or worse than the
//!    locked one? A flatter prior spreads the same simulations thinner, so
//!    at low budget it may well be worse; that is a cost, not a refutation.
//! 3. Temperature alone, holding c at the default, to separate the two knobs.

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsConfig, Net};
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::Agent;

fn report(
    title: &str,
    net: &Net,
    a: MctsConfig,
    b: MctsConfig,
    pairs: u32,
    seed: u64,
    cores: usize,
) {
    let (na, nb) = (net, net);
    let res = run_match_parallel(
        move || Box::new(NetMctsAgent::new(a, na)) as Box<dyn Agent>,
        move || Box::new(NetMctsAgent::new(b, nb)) as Box<dyn Agent>,
        pairs,
        seed,
        &Kingdoms::Random,
        cores,
    );
    let sigma = (res.win_rate_a() - 0.5).abs() / res.stderr().max(1e-9);
    println!("{title}");
    println!("  {res}");
    println!("  {sigma:.1} standard errors from even\n");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "models/net.bin".into());
    let pairs: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let net = Net::load(&path).expect("load network");
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // The setting the sweep pointed at.
    let unlocked = |w: u32, i: u32| MctsConfig {
        worlds: w,
        iterations: i,
        exploration: 2.5,
        prior_temperature: 4.0,
        ..Default::default()
    };
    let locked = |w: u32, i: u32| MctsConfig {
        worlds: w,
        iterations: i,
        exploration: 2.5,
        prior_temperature: 1.0,
        ..Default::default()
    };

    report(
        "1. unlocked 16x800 vs unlocked 4x200 (16x the budget, both temp 4.0):",
        &net,
        unlocked(16, 800),
        unlocked(4, 200),
        pairs,
        0xBEEF,
        cores,
    );

    report(
        "2. unlocked 8x400 vs locked 8x400 (equal budget, temp 4.0 vs 1.0):",
        &net,
        unlocked(8, 400),
        locked(8, 400),
        pairs,
        0xC0FFEE,
        cores,
    );

    report(
        "3. unlocked 16x800 vs locked 16x800 (equal high budget):",
        &net,
        unlocked(16, 800),
        locked(16, 800),
        pairs,
        0xD00D,
        cores,
    );

    println!("Reference under the locked prior: 16x budget won 50.42% +/- 4.56.");
}
