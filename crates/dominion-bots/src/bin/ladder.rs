//! Round-robin benchmark over the built-in strategies.
//!
//! Usage: `cargo run --release --bin ladder -- [pairs_per_matchup]`
//!
//! Every pairing is played on random kingdoms containing the cards both
//! strategies need, so the numbers reflect strategy strength rather than luck
//! of the kingdom draw.

use dominion_bots::buy::{ladder, required_kingdom, MenuBot};
use dominion_bots::match_runner::{run_match, Kingdoms};
use dominion_bots::policy::{HeuristicBot, RandomAgent};
use dominion_bots::Agent;

fn main() {
    let pairs: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let menus = ladder();
    println!(
        "Round robin: {} strategies, {} games per pairing\n",
        menus.len(),
        pairs * 2
    );

    let mut totals = vec![0.0f32; menus.len()];
    let mut counts = vec![0u32; menus.len()];

    for i in 0..menus.len() {
        for j in (i + 1)..menus.len() {
            let mut a = MenuBot::new(menus[i].clone());
            let mut b = MenuBot::new(menus[j].clone());
            let mut must = required_kingdom(&menus[i]);
            must.extend(required_kingdom(&menus[j]));
            let res = run_match(&mut a, &mut b, pairs, 0xC0FFEE, &Kingdoms::RandomWith(must));
            println!("{res}");
            totals[i] += res.win_rate_a();
            totals[j] += 1.0 - res.win_rate_a();
            counts[i] += 1;
            counts[j] += 1;
        }
    }

    // The menu-free heuristic, measured against the whole ladder. This is the
    // baseline the search agents have to beat to have earned their cost.
    println!();
    for menu in &menus {
        let mut h = HeuristicBot;
        let mut foe = MenuBot::new(menu.clone());
        let must = required_kingdom(menu);
        let res = run_match(&mut h, &mut foe, pairs, 0xC0FFEE, &Kingdoms::RandomWith(must));
        println!("{res}");
    }

    // Sanity floor: the weakest menu should still crush random play.
    let mut bm = MenuBot::new(menus[0].clone());
    let mut rand = RandomAgent::new(7);
    let res = run_match(&mut bm, &mut rand, pairs.min(200), 1234, &Kingdoms::Random);
    println!("\n{res}");

    let mut table: Vec<(String, f32)> = menus
        .iter()
        .enumerate()
        .map(|(i, m)| (m.name.clone(), totals[i] / counts[i].max(1) as f32))
        .collect();
    table.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nAverage win rate across all pairings:");
    for (name, wr) in table {
        println!("  {name:<14} {:>6.2}%", wr * 100.0);
    }
    let _ = bm.name();
}
