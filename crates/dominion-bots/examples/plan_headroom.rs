//! How much is there to gain from committing to a plan at all?
//!
//! Adding engine cards to the per-card ranking one at a time made the
//! heuristic worse — Chapel alone cost fourteen points — because every card in
//! an engine is a bad buy until the others are present. The conclusion drawn
//! from that was that the buy policy should commit to a *plan* the kingdom
//! supports and buy consistently for it, the way `MenuBot` does.
//!
//! That is a large piece of work, and this project has spent a day learning
//! that plausible next steps measure zero. So measure the headroom first.
//!
//! For each random kingdom, every hand-written plan the kingdom actually
//! supports is played against the current heuristic on that same kingdom. The
//! oracle — the best plan per kingdom, chosen with hindsight — is an upper
//! bound on what perfect plan selection could buy. If the oracle barely beats
//! the heuristic, no plan-selection machinery is worth building, however well
//! it is implemented. If it wins clearly, the gap is real and this also names
//! which plans are worth having.
//!
//! Two things it deliberately does not do. It does not let a plan be chosen on
//! a kingdom missing its cards, because that is not a choice available in a
//! real game. And it reports how often each plan wins its kingdom, not just
//! the average, because one plan dominating everywhere and ten plans each
//! winning somewhere call for completely different machinery.

use std::collections::HashMap;

use dominion_bots::buy::*;
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::policy::HeuristicBot;
use dominion_bots::Agent;
use dominion_core::{Card, Game, Rng};

fn r(card: Card) -> BuyRule {
    BuyRule::new(card)
}

/// Every plan on offer, including the engine lines the AI never builds.
fn plans() -> Vec<BuyMenu> {
    use Card::*;
    let mut v = ladder();
    v.push(chapel_engine());
    v.push(BuyMenu::new(
        "Throne/Vassal",
        vec![
            r(Province),
            r(Gold),
            r(Chapel).at_most(1).while_provinces_above(6),
            r(ThroneRoom).at_most(2),
            r(Vassal).at_most(4),
            r(Festival).at_most(2),
            r(Duchy).when_provinces_at_most(4),
            r(Silver),
        ],
    ));
    v.push(BuyMenu::new(
        "Lab+Money",
        vec![
            r(Province),
            r(Gold),
            r(Laboratory).at_most(4),
            r(Duchy).when_provinces_at_most(4),
            r(Silver),
        ],
    ));
    v.push(BuyMenu::new(
        "Festival+Lab",
        vec![
            r(Province),
            r(Gold),
            r(Laboratory).at_most(3),
            r(Festival).at_most(2),
            r(Market).at_most(2),
            r(Duchy).when_provinces_at_most(4),
            r(Silver),
        ],
    ));
    v
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let kingdoms: u32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(60);
    let pairs: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let all = plans();
    let mut wins: HashMap<String, u32> = HashMap::new();
    let mut oracle_total = 0.0f64;
    let mut best_single: HashMap<String, (f64, u32)> = HashMap::new();
    let mut evaluated = 0u32;

    for k in 0..kingdoms {
        let kingdom = Game::random_kingdom(&mut Rng::new(k as u64 * 7919 + 11));
        // Only plans this kingdom can actually support are candidates.
        let usable: Vec<&BuyMenu> = all
            .iter()
            .filter(|m| required_kingdom(m).iter().all(|c| kingdom.contains(c)))
            .collect();
        if usable.is_empty() {
            continue;
        }
        evaluated += 1;

        let mut best = (String::new(), 0.0f64);
        for m in &usable {
            let menu = (*m).clone();
            let res = run_match_parallel(
                move || Box::new(MenuBot::new(menu.clone())) as Box<dyn Agent>,
                || Box::new(HeuristicBot) as Box<dyn Agent>,
                pairs,
                0xBEEF + k as u64,
                &Kingdoms::Fixed(kingdom.clone()),
                cores,
            );
            let wr = res.win_rate_a() as f64;
            let e = best_single.entry(m.name.clone()).or_insert((0.0, 0));
            e.0 += wr;
            e.1 += 1;
            if wr > best.1 {
                best = (m.name.clone(), wr);
            }
        }
        oracle_total += best.1;
        *wins.entry(best.0).or_insert(0) += 1;
    }

    println!(
        "{evaluated} kingdoms, {} games per plan per kingdom, all against the heuristic\n",
        pairs * 2
    );
    println!(
        "oracle (best supported plan per kingdom, chosen with hindsight): {:.2}%",
        100.0 * oracle_total / evaluated.max(1) as f64
    );
    println!("the heuristic is 50% here by construction — it is the opponent\n");

    println!("{:<16} {:>10} {:>16}", "plan", "avg", "kingdoms it won");
    let mut rows: Vec<(String, f64, u32)> = best_single
        .into_iter()
        .map(|(k, (sum, n))| {
            let w = *wins.get(&k).unwrap_or(&0);
            (k, 100.0 * sum / n.max(1) as f64, w)
        })
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (name, avg, won) in rows {
        println!("{name:<16} {avg:>9.2}% {won:>16}");
    }

    println!(
        "\nThe oracle is an upper bound: it picks with hindsight, per kingdom, from\n\
         plans that are already hand-written. Real selection can only do worse."
    );
}
