//! Machine check and strength benchmark in one command.
//!
//! `cargo run --release --bin bench` reports what this machine can do (cores,
//! engine throughput, search latency) and then plays the search agent against
//! the heuristic ladder.
//!
//! Optional args: `bench [games] [worlds] [iterations]`

use std::time::Instant;

use dominion_ai::{MctsAgent, MctsConfig};
use dominion_bots::buy::{big_money_smithy, double_witch, required_kingdom, MenuBot};
use dominion_bots::match_runner::{play_game, run_match, Kingdoms};
use dominion_bots::Agent;
use dominion_core::{Game, Rng};

fn main() {
    let mut args = std::env::args().skip(1);
    let pairs: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(25);
    let worlds: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let iterations: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(300);

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("== machine ==");
    println!("logical cores: {cores}");

    // Engine throughput with heuristic agents, single-threaded.
    let mut rng = Rng::new(1);
    let kingdom = Game::random_kingdom(&mut rng);
    let mut a = MenuBot::new(big_money_smithy());
    let mut b = MenuBot::new(double_witch());
    let n = 3000;
    let t = Instant::now();
    for i in 0..n {
        play_game(&kingdom, &mut [&mut a, &mut b], i as u64);
    }
    let per_core = n as f64 / t.elapsed().as_secs_f64();
    println!("engine throughput: {per_core:.0} games/s/core, ~{:.0} games/s across {cores} cores",
             per_core * cores as f64);

    // Search latency at the requested budget.
    let cfg = MctsConfig {
        worlds,
        iterations,
        ..Default::default()
    };
    let mut searcher = MctsAgent::new(cfg);
    let mut opp = MenuBot::new(big_money_smithy());
    let t = Instant::now();
    let (_, turns) = play_game(&kingdom, &mut [&mut searcher, &mut opp], 7);
    let secs = t.elapsed().as_secs_f64();
    println!(
        "search: {worlds} worlds x {iterations} iters -> {secs:.1}s per game ({turns} turns)"
    );
    println!(
        "estimated self-play rate: {:.1} games/s across {cores} cores",
        cores as f64 / secs
    );

    println!("\n== strength ==");
    for menu in [big_money_smithy(), double_witch()] {
        let mut searcher = MctsAgent::new(cfg);
        let mut foe = MenuBot::new(menu.clone());
        let must = required_kingdom(&menu);
        let res = run_match(
            &mut searcher,
            &mut foe,
            pairs,
            0xBEEF,
            &Kingdoms::RandomWith(must),
        );
        println!("{res}");
    }
    let _ = opp.name();
}
