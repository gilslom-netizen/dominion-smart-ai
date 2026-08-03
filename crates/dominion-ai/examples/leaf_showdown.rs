//! Does the better-calibrated leaf estimate actually win games?
//!
//! `value_calibration` measured the heuristic rollout as substantially better
//! calibrated than the trained value head over 1,641 positions — Brier 0.1301
//! against 0.1599, correlation 0.639 against 0.572, ahead in every third of
//! the game. That is a property of the estimator, not a result about play,
//! and the two come apart often enough to be worth separating.
//!
//! Two comparisons, because the rollout is much more expensive per leaf and
//! the interesting question depends on what is scarce:
//!
//! * **Game-matched** — same worlds and iterations on both sides. Answers
//!   "is the rollout a better thing to back up?", ignoring what it costs.
//! * **Compute-matched** — the value head gets extra iterations to spend the
//!   same wall-clock. Answers the question that actually decides the default,
//!   since a search is always budgeted in time, not in simulations.
//!
//! The multiplier for the compute-matched arm is measured here rather than
//! assumed, because guessing it is how a comparison ends up confounded.

use std::time::Instant;

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsConfig, Net};
use dominion_bots::match_runner::{play_game, run_match_parallel, Kingdoms};
use dominion_bots::Agent;
use dominion_core::{Game, Rng};

fn cfg_for(worlds: u32, iterations: u32, use_value_head: bool) -> MctsConfig {
    MctsConfig {
        worlds,
        iterations,
        use_value_head,
        ..Default::default()
    }
}

/// Seconds for one self-play game at this configuration.
fn time_one(net: &Net, cfg: MctsConfig) -> f64 {
    let kingdom = Game::random_kingdom(&mut Rng::new(1));
    let t = Instant::now();
    let mut a = NetMctsAgent::new(cfg, net);
    let mut b = NetMctsAgent::new(cfg, net);
    play_game(&kingdom, &mut [&mut a, &mut b], 7);
    t.elapsed().as_secs_f64()
}

fn report(title: &str, net: &Net, a: MctsConfig, b: MctsConfig, pairs: u32, seed: u64, cores: usize) {
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
    let path = args.first().cloned().unwrap_or_else(|| "models/net.bin".into());
    let pairs: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let net = Net::load(&path).expect("load network");
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    let value_head = cfg_for(8, 400, true);
    let rollout = cfg_for(8, 400, false);

    let t_value = time_one(&net, value_head);
    let t_roll = time_one(&net, rollout);
    println!(
        "cost per game at 8x400: value head {t_value:.1}s, rollout {t_roll:.1}s ({:.1}x)\n",
        t_roll / t_value.max(1e-9)
    );

    report(
        "1. game-matched — rollout 8x400 vs value head 8x400:",
        &net,
        rollout,
        value_head,
        pairs,
        0xBEEF,
        cores,
    );

    // Give the value head the iterations it can afford in the rollout's time.
    let scaled = ((400.0 * t_roll / t_value.max(1e-9)).round() as u32).clamp(400, 40_000);
    report(
        &format!(
            "2. compute-matched — rollout 8x400 vs value head 8x{scaled} (same wall clock):"
        ),
        &net,
        rollout,
        cfg_for(8, scaled, true),
        pairs,
        0xC0FFEE,
        cores,
    );

    println!(
        "Same network on both sides throughout, so the leaf estimator is the\n\
         only variable. Calibration reference: rollout Brier 0.1301 vs the\n\
         value head's 0.1599 over 1,641 positions."
    );
}
