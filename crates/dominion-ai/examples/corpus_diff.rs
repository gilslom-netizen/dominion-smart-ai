//! Do two corpora that cover the same games actually play them differently?
//!
//! `corpus_overlap` identifies a game by `(seed, kingdom)`, which is the right
//! key for spotting wasted effort across machines but the wrong one for a
//! paired experiment. Two runs deliberately given the same seed — so they
//! cover identical kingdoms and differ only in how the search evaluates leaves
//! — report as 100% overlapping while containing entirely different play.
//!
//! This compares the games themselves: for each `(seed, kingdom)` present in
//! both files, how often the move sequences diverge, where they first diverge,
//! and how far apart the recorded policies are. If a change to the search was
//! supposed to alter play and this reports near-zero divergence, the change
//! did not take effect and any downstream result would be measuring nothing.

use std::collections::HashMap;

use dominion_ai::compact::{game_id, read_games};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: corpus_diff <a.gamelog> <b.gamelog>");
        std::process::exit(2);
    }

    let a = read_games(&args[0]).expect("read a");
    let b = read_games(&args[1]).expect("read b");
    let index: HashMap<_, _> = b.iter().map(|g| (game_id(g), g)).collect();

    let (mut paired, mut same_moves, mut same_len) = (0u32, 0u32, 0u32);
    let mut first_div = Vec::new();
    let mut policy_l1 = 0.0f64;
    let mut policy_n = 0u32;

    for ga in &a {
        let Some(gb) = index.get(&game_id(ga)) else {
            continue;
        };
        paired += 1;
        if ga.moves.len() == gb.moves.len() {
            same_len += 1;
        }
        if ga.moves == gb.moves {
            same_moves += 1;
        } else {
            let at = ga
                .moves
                .iter()
                .zip(&gb.moves)
                .position(|(x, y)| x != y)
                .unwrap_or(ga.moves.len().min(gb.moves.len()));
            first_div.push(at);
        }

        // Policy distance on the decisions the two share by index, as a
        // second signal: play can coincide while the search's confidence
        // moved, and that alone changes what the network trains on.
        for (da, db) in ga.decisions.iter().zip(&gb.decisions) {
            if da.ply != db.ply {
                break;
            }
            let pa: HashMap<usize, f32> =
                da.policy.iter().map(|(m, p)| (m.index(), *p)).collect();
            let mut l1 = 0.0f64;
            for (m, p) in &db.policy {
                let other = pa.get(&m.index()).copied().unwrap_or(0.0);
                l1 += (p - other).abs() as f64;
            }
            policy_l1 += l1;
            policy_n += 1;
        }
    }

    println!("{} games in a, {} in b, {paired} paired by (seed, kingdom)", a.len(), b.len());
    if paired == 0 {
        println!("\nNothing to compare — these corpora cover different games.");
        return;
    }
    println!(
        "identical move sequences: {same_moves} of {paired} ({:.1}%)",
        100.0 * same_moves as f64 / paired as f64
    );
    println!(
        "identical game length:    {same_len} of {paired} ({:.1}%)",
        100.0 * same_len as f64 / paired as f64
    );
    if !first_div.is_empty() {
        first_div.sort_unstable();
        let med = first_div[first_div.len() / 2];
        println!(
            "of the {} that diverge, first difference at move {med} (median)",
            first_div.len()
        );
    }
    if policy_n > 0 {
        println!(
            "mean L1 distance between recorded policies: {:.3} over {policy_n} shared decisions",
            policy_l1 / policy_n as f64
        );
    }
    println!(
        "\n(L1 ranges 0 to 2. Near 0 with near-100% identical moves means the\n\
         two runs are the same experiment and any comparison between them is\n\
         measuring noise.)"
    );
}
