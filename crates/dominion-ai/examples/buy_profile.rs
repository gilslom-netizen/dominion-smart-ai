//! What does the AI actually buy on a kingdom built for an engine?
//!
//! Everything measured so far says the AI prefers money, but "prefers" is a
//! guess about a mechanism. This records the fact: put Chapel, Festival,
//! Throne Room, Vassal, Village and Laboratory in the supply, hand the AI its
//! normal search, and count every card it buys.
//!
//! The distinction that matters is between *choosing* money and *never
//! considering* anything else. `gain_preference` ranks unnamed Actions at
//! `300 + cost`, below Silver's 700, so those cards lose every comparison at
//! the prior — and the search allocates its budget by the prior. A histogram
//! separates the two: a deck that buys a few Throne Rooms and finds them bad
//! looks nothing like a deck that has never bought one.
//!
//! Also reports what the search *considered*: how often an engine card was a
//! legal option at all, against how often it was taken.

use std::collections::HashMap;

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsConfig, Net};
use dominion_bots::policy::HeuristicBot;
use dominion_bots::Agent;
use dominion_core::{Card, Game, Move, Rng};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "models/net.bin".into());
    let games: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let iterations: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);
    let net = Net::load(&path).expect("load network");

    use Card::*;
    // Every piece the Throne Room / Vassal engine needs, plus room to breathe.
    let kingdom = vec![
        Chapel, Festival, ThroneRoom, Vassal, Village, Laboratory, Market, Smithy, Militia, Moat,
    ];
    let engine_cards = [Chapel, Festival, ThroneRoom, Vassal, Village, Laboratory];

    let cfg = MctsConfig {
        worlds: 8,
        iterations,
        ..Default::default()
    };

    let mut bought: HashMap<Card, u32> = HashMap::new();
    let mut offered: HashMap<Card, u32> = HashMap::new();
    let mut turns_total = 0u32;

    for g in 0..games {
        let mut game = Game::new(&kingdom, 2, g as u64 * 31 + 7).unwrap();
        let mut ai = NetMctsAgent::new(cfg, &net);
        let mut opp = HeuristicBot;
        let ai_seat = (g % 2) as usize;

        let mut guard = 0u32;
        while !game.is_over() && guard < 4000 {
            guard += 1;
            let d = game.decision().expect("live game").clone();
            let mv = if d.player == ai_seat {
                // Count what was on the table before choosing, so "never
                // bought" can be told apart from "never offered".
                for o in &d.options {
                    if let Move::Buy(c) = o {
                        *offered.entry(*c).or_insert(0) += 1;
                    }
                }
                let mv = ai.decide(&game.state, &d);
                if let Move::Buy(c) = mv {
                    *bought.entry(c).or_insert(0) += 1;
                }
                mv
            } else {
                opp.decide(&game.state, &d)
            };
            game.apply(mv).expect("legal move");
        }
        turns_total += guard;
    }

    println!(
        "{games} games on a kingdom containing every engine piece, {}x{} search\n",
        cfg.worlds, cfg.iterations
    );
    println!("{:<14} {:>8} {:>10} {:>10}", "card", "bought", "offered", "taken");
    let mut rows: Vec<(Card, u32, u32)> = kingdom
        .iter()
        .chain([Copper, Silver, Gold, Estate, Duchy, Province].iter())
        .map(|&c| {
            (
                c,
                *bought.get(&c).unwrap_or(&0),
                *offered.get(&c).unwrap_or(&0),
            )
        })
        .collect();
    rows.sort_by_key(|(_, b, _)| std::cmp::Reverse(*b));
    for (c, b, o) in rows {
        let rate = if o > 0 {
            format!("{:.1}%", 100.0 * b as f64 / o as f64)
        } else {
            "—".into()
        };
        println!("{:<14} {b:>8} {o:>10} {rate:>10}", format!("{c}"));
    }

    let engine_bought: u32 = engine_cards
        .iter()
        .map(|c| *bought.get(c).unwrap_or(&0))
        .sum();
    let total_bought: u32 = bought.values().sum();
    println!(
        "\nengine pieces bought: {engine_bought} of {total_bought} purchases ({:.1}%)",
        100.0 * engine_bought as f64 / total_bought.max(1) as f64
    );
    println!("(avg {:.0} decisions per game)", turns_total as f64 / games as f64);
}
