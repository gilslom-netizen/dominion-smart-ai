//! Read a game a human actually played and say who built what.
//!
//! A `GameLog` is one interleaved move list, so "what did each side buy" is
//! not in the file — it has to be replayed to be recovered. The engine is
//! deterministic given the kingdom and seed, so replaying each prefix and
//! asking whose decision it was attributes every move exactly.
//!
//! This exists because self-play cannot produce the interesting cases. It can
//! only ever show the AI lines the AI already plays; a human who beats it
//! shows it one it does not. Handles a file of several games separated by
//! `# game ...` lines, which is what the web client exports.

use dominion_core::{Card, GameLog, Move};

fn tally(mut cards: Vec<Card>) -> String {
    cards.sort_by_key(|c| (std::cmp::Reverse(c.cost()), format!("{c}")));
    let mut out: Vec<(Card, usize)> = Vec::new();
    for c in cards {
        match out.iter_mut().find(|(x, _)| *x == c) {
            Some((_, n)) => *n += 1,
            None => out.push((c, 1)),
        }
    }
    out.iter()
        .map(|(c, n)| if *n == 1 { format!("{c}") } else { format!("{c}×{n}") })
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: analyse_log <file>");
    let text = std::fs::read_to_string(&path).expect("read log");

    // The export separates games with a comment line; a single game has none.
    let chunks: Vec<String> = if text.contains("# game ") {
        text.split("# game ")
            .filter(|c| c.contains("moves:"))
            .map(|c| c.to_string())
            .collect()
    } else {
        vec![text]
    };

    for (gi, chunk) in chunks.iter().enumerate() {
        let body = chunk[chunk.find("kingdom:").unwrap_or(0)..].to_string();
        let log = match GameLog::from_text(&body) {
            Ok(l) => l,
            Err(e) => {
                println!("game {}: could not parse ({e:?})\n", gi + 1);
                continue;
            }
        };

        let mut buys: [Vec<Card>; 2] = [Vec::new(), Vec::new()];
        let mut trashed: [Vec<Card>; 2] = [Vec::new(), Vec::new()];
        let mut first_province: [Option<usize>; 2] = [None, None];

        // Attribute each move by replaying up to it and asking whose turn it
        // was. Quadratic, and irrelevant at a few hundred moves.
        for i in 0..log.moves.len() {
            let Ok(g) = log.replay_prefix(i) else { break };
            let Some(d) = g.decision() else { break };
            let p = d.player;
            match log.moves[i] {
                Move::Buy(c) => {
                    buys[p].push(c);
                    if c == Card::Province && first_province[p].is_none() {
                        first_province[p] = Some(i);
                    }
                }
                // Chapel is the only mass-trasher here, and it is the whole
                // question, so its picks are worth separating out.
                Move::Select(c) if d.ctx == dominion_core::Ctx::ChapelTrash => trashed[p].push(c),
                _ => {}
            }
        }

        // A log written before the undo fix can diverge partway: undo popped
        // one history entry while "play all Treasures" had pushed several, so
        // the file describes a game that never happened from that point on.
        // Report how far it got rather than refusing to say anything.
        let (end, truncated) = match log.replay() {
            Ok(g) => (g, None),
            Err(_) => {
                let mut last = 0;
                for i in (0..log.moves.len()).rev() {
                    if log.replay_prefix(i).is_ok() {
                        last = i;
                        break;
                    }
                }
                (
                    log.replay_prefix(last).expect("prefix replays"),
                    Some(last),
                )
            }
        };
        if let Some(at) = truncated {
            println!(
                "note: this log stops replaying at move {at} of {} — recorded before \
                 the undo fix, so everything after that point is unreadable.",
                log.moves.len()
            );
        }
        // A log that replays without erroring is not necessarily the game that
        // was played. The undo bug left extra moves behind, and if those moves
        // happen to be legal in the shifted position the replay succeeds and
        // silently describes a different game — with every later move
        // attributed to the wrong player. A game that does not end in a real
        // ending is the tell.
        let ended = end.is_over();
        let provinces_gone = end.state.supply_of(Card::Province) == 0;
        let empty_piles = (0..dominion_core::NUM_CARDS)
            .filter(|&i| end.state.in_supply[i] && end.state.supply[i] == 0)
            .count();
        if !ended {
            println!(
                "WARNING: this log replays but the game never ends — {} Provinces left, \
                 {empty_piles} empty piles. Recorded before the undo fix, so the move list \
                 is probably not the game that was played, and per-player attribution \
                 below is unreliable.",
                end.state.supply_of(Card::Province)
            );
        }

        let scores = end.state.scores();

        println!("=== game {} — seed {} ===", gi + 1, log.seed);
        println!("kingdom: {}", tally(log.kingdom.clone()));
        for p in 0..2 {
            println!(
                "\nplayer {p}  (final {} VP, {} cards)",
                scores[p],
                end.state.players[p].total_cards()
            );
            println!("  bought:  {}", tally(buys[p].clone()));
            if !trashed[p].is_empty() {
                println!("  chapelled away: {}", tally(trashed[p].clone()));
            }
            match first_province[p] {
                Some(i) => println!("  first Province at move {i} of {}", log.moves.len()),
                None => println!("  never bought a Province"),
            }
        }
        println!(
            "\nended properly: {ended}  (Provinces gone: {provinces_gone}, empty piles: {empty_piles})"
        );
        let winner = if scores[0] > scores[1] {
            "player 0"
        } else if scores[1] > scores[0] {
            "player 1"
        } else {
            "nobody — tied"
        };
        println!("\nwinner: {winner}\n");
    }
}
