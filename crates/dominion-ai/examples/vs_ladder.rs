//! Is it actually better than Big Money?
//!
//! That question deserves a number rather than an opinion, and the honest
//! version of it is not one matchup. Big Money is a family: plain BM, BM with
//! a single terminal, BM with an attack, and the thin-deck Chapel variant that
//! plays nothing like the others. A bot can beat one of them by luck of the
//! kingdom and lose to the next.
//!
//! So this plays the network against every menu on the ladder, on kingdoms
//! that contain whatever each menu needs, and reports each separately. The
//! per-menu breakdown is the point: an average hides the case where the AI
//! beats three money decks comfortably and loses to the one that trashes.

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsConfig, Net};
use dominion_bots::buy::{ladder, required_kingdom, MenuBot};
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::Agent;

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

    // The cheaper search setting, so every menu can be measured at a sample
    // size worth reading rather than one menu at a sample size that is not.
    let cfg = MctsConfig {
        worlds: 4,
        iterations: 200,
        ..Default::default()
    };

    println!("{path} at {}x{}\n", cfg.worlds, cfg.iterations);
    println!("{:<18} {:>9} {:>10} {:>8}", "opponent", "win rate", "±", "sigma");

    let mut total = 0.0f64;
    let mut n = 0u32;
    for menu in ladder() {
        let name = menu.name.clone();
        let must = required_kingdom(&menu);
        let net_ref = &net;
        let res = run_match_parallel(
            move || Box::new(NetMctsAgent::new(cfg, net_ref)) as Box<dyn Agent>,
            move || Box::new(MenuBot::new(menu.clone())) as Box<dyn Agent>,
            pairs,
            0xBEEF,
            &Kingdoms::RandomWith(must),
            cores,
        );
        let wr = res.win_rate_a();
        let sigma = (wr - 0.5).abs() / res.stderr().max(1e-9);
        println!(
            "{:<18} {:>8.2}% {:>9.2} {:>7.1}",
            name,
            100.0 * wr,
            100.0 * res.stderr(),
            sigma
        );
        total += wr as f64;
        n += 1;
    }

    println!(
        "\naverage across {n} menus: {:.2}%",
        100.0 * total / n.max(1) as f64
    );
    println!(
        "\nFor reference, the hand-written heuristic averages 64.1% against this\n\
         same ladder. Anything the network adds is on top of that."
    );
}
