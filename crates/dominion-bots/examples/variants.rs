//! Is an engine actually competitive in Base 2E, or is money simply right?
//!
//! The search prefers money on every kingdom. Before treating that as a flaw
//! worth engineering around, it is worth knowing whether it is a flaw at all:
//! Base 2E is widely held to be a set where money strategies are strong and
//! engines are deliberately weak, so the preference may be correct.
//!
//! This plays several engine-leaning menus against the same six-menu ladder
//! the heuristic is scored on, so the numbers are directly comparable to its
//! 64.1%.
//!
//! The first entry is the menu that produced a catastrophic 14.65% and nearly
//! led to the conclusion that engines are hopeless. It has no Gold in it. The
//! second is the same list with Gold added and is worth 46% — the difference
//! measured the menu, not the strategy, which is the trap this file exists to
//! avoid falling into twice.
//!
//! A fixed priority list still pilots an engine badly: it cannot buy
//! conditionally on what it has already assembled, and the shared trash policy
//! keeps roughly $4 of coin rather than thinning as hard as a real Chapel deck
//! would. Every number here is therefore a lower bound.

use dominion_bots::buy::*;
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::Agent;
use dominion_core::Card::{self, *};
fn r(card: Card) -> BuyRule { BuyRule::new(card) }

fn main() {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let variants: Vec<BuyMenu> = vec![
        BuyMenu::new("Eng-noGold", vec![
            r(Province), r(Chapel).at_most(1).while_provinces_above(6),
            r(Festival).at_most(3), r(Laboratory).at_most(3), r(Market).at_most(2),
            r(ThroneRoom).at_most(1), r(Vassal).at_most(2),
            r(Duchy).when_provinces_at_most(4), r(Silver)]),
        BuyMenu::new("Eng+Gold", vec![
            r(Province), r(Gold),
            r(Chapel).at_most(1).while_provinces_above(6),
            r(Laboratory).at_most(3), r(Festival).at_most(2), r(Market).at_most(2),
            r(Duchy).when_provinces_at_most(4), r(Silver)]),
        BuyMenu::new("Lab+Money", vec![
            r(Province), r(Gold), r(Laboratory).at_most(4),
            r(Duchy).when_provinces_at_most(4), r(Silver)]),
        BuyMenu::new("Festival+Lab", vec![
            r(Province), r(Gold), r(Laboratory).at_most(3), r(Festival).at_most(2),
            r(Market).at_most(2), r(Duchy).when_provinces_at_most(4), r(Silver)]),
        BuyMenu::new("NoChapelEngine", vec![
            r(Province), r(Gold), r(Village).at_most(3), r(Laboratory).at_most(3),
            r(Smithy).at_most(2), r(Market).at_most(2),
            r(Duchy).when_provinces_at_most(4), r(Silver)]),
    ];
    println!("{:<16} {:>9}   (average win rate vs the 6-menu ladder)", "menu", "avg");
    for v in &variants {
        let mut total = 0.0f64;
        for opp in ladder() {
            let mut must = required_kingdom(v);
            must.extend(required_kingdom(&opp));
            let a = v.clone();
            let res = run_match_parallel(
                move || Box::new(MenuBot::new(a.clone())) as Box<dyn Agent>,
                move || Box::new(MenuBot::new(opp.clone())) as Box<dyn Agent>,
                150, 0xBEEF, &Kingdoms::RandomWith(must), cores);
            total += res.win_rate_a() as f64;
        }
        println!("{:<16} {:>8.2}%", v.name, 100.0 * total / 6.0);
    }
    println!("\nreference: the hand-written heuristic averages 64.1% against this ladder");
}
