//! Machine check and strength benchmark in one command.
//!
//! `cargo run --release --bin bench` reports what this machine can do (cores,
//! engine throughput, search cost) and then plays the search agent against the
//! heuristic it is built on.
//!
//! The headline number is MCTS vs Heuristic. The search uses the heuristic as
//! its prior *and* as its rollout policy, so anything above 50% there is
//! strength the search itself added, not a better hand-written strategy.
//!
//! Args: `bench [pairs] [worlds] [iterations]`

use std::time::Instant;

use dominion_ai::{MctsAgent, MctsConfig};
use dominion_bots::buy::{double_witch, required_kingdom, MenuBot};
use dominion_bots::match_runner::{play_game, run_match_parallel, Kingdoms};
use dominion_bots::policy::HeuristicBot;
use dominion_bots::Agent;
use dominion_core::{Game, Rng};

fn main() {
    let mut args = std::env::args().skip(1);
    let pairs: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);
    let worlds: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let iterations: u32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(400);

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("== machine ==");
    println!("logical cores: {cores}");

    let mut rng = Rng::new(1);
    let kingdom = Game::random_kingdom(&mut rng);
    let n = 3000;
    let t = Instant::now();
    for i in 0..n {
        let mut a = HeuristicBot;
        let mut b = HeuristicBot;
        play_game(&kingdom, &mut [&mut a, &mut b], i as u64);
    }
    let per_core = n as f64 / t.elapsed().as_secs_f64();
    println!(
        "engine throughput: {per_core:.0} games/s/core, ~{:.0} games/s across {cores} cores",
        per_core * cores as f64
    );

    let cfg = MctsConfig {
        worlds,
        iterations,
        ..Default::default()
    };
    let t = Instant::now();
    let mut searcher = MctsAgent::new(cfg);
    let mut opp = HeuristicBot;
    let (_, turns) = play_game(&kingdom, &mut [&mut searcher, &mut opp], 7);
    let secs = t.elapsed().as_secs_f64();
    println!("search: {worlds} worlds x {iterations} iters -> {secs:.1}s per game ({turns} turns)");

    // Two earlier estimates of the self-play rate were reported ~2x optimistic
    // against what production runs actually measured, for two reasons that
    // both live in the extrapolation rather than the timing:
    //
    //   1. the timed game above searches on *one* side and lets the heuristic
    //      play the other, while self-play searches on both — roughly double
    //      the work per game; and
    //   2. `cores / secs` assumes throughput scales linearly with cores, which
    //      it does not once memory bandwidth and SMT siblings are in play.
    //
    // So measure the thing directly instead of extrapolating to it: run one
    // search-vs-search game on every core at once and time the wall clock.
    // Costs about as long as a single game and needs no scaling assumption.
    let t = Instant::now();
    std::thread::scope(|s| {
        for c in 0..cores {
            let kingdom = &kingdom;
            s.spawn(move || {
                let cfg = MctsConfig {
                    seed: 0x5EED ^ (c as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    ..cfg
                };
                let mut a = MctsAgent::new(cfg);
                let mut b = MctsAgent::new(cfg);
                play_game(kingdom, &mut [&mut a, &mut b], 7 + c as u64);
            });
        }
    });
    let wall = t.elapsed().as_secs_f64();
    println!(
        "measured self-play rate: {:.2} games/s ({cores} concurrent search-vs-search games in {wall:.1}s)",
        cores as f64 / wall
    );
    println!(
        "  (naive {:.2} games/s if you assume one-sided search and linear scaling)",
        cores as f64 / secs
    );
    println!(
        "  net-guided self-play (--net) is slower still: the prior costs a forward pass per node."
    );

    println!("\n== strength ({} games per matchup) ==", pairs * 2);

    // The one that matters: search against its own rollout policy.
    let res = run_match_parallel(
        || Box::new(MctsAgent::new(cfg)) as Box<dyn Agent>,
        || Box::new(HeuristicBot) as Box<dyn Agent>,
        pairs,
        0xBEEF,
        &Kingdoms::Random,
        cores,
    );
    println!("{res}");

    // And against the strongest hand-written menu, as a cross-check.
    let menu = double_witch();
    let res = run_match_parallel(
        || Box::new(MctsAgent::new(cfg)) as Box<dyn Agent>,
        move || Box::new(MenuBot::new(double_witch())) as Box<dyn Agent>,
        pairs,
        0xBEEF,
        &Kingdoms::RandomWith(required_kingdom(&menu)),
        cores,
    );
    println!("{res}");
}
