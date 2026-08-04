//! Play a game against the AI in the terminal.
//!
//! ```text
//! cargo run --release --bin play                      # you vs the network
//! cargo run --release --bin play -- --seed 42         # a reproducible game
//! cargo run --release --bin play -- --iterations 1500 # give it more thinking
//! cargo run --release --bin play -- --heuristic       # no network, search only
//! ```
//!
//! Every move you both make is appended to a `GameLog`, written out at the end.
//! That log is the same format `bin/advise` reads, so any position from the
//! game can be handed back to the AI afterwards to ask what it would have done
//! — which is more informative than the result on its own.

use std::io::{BufRead, Write};

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsAgent, MctsConfig, Net};
use dominion_bots::Agent;
use dominion_core::{Card, Ctx, Decision, Game, GameLog, GameState, Move, Rng};

const USAGE: &str = "\
play a game of Dominion against the AI

usage: play [options]

  --seed <n>         fix the shuffle and kingdom, for a reproducible game
  --net <path>       network to use (default models/net.bin)
  --heuristic        no network: the search steered by the hand-written prior
  --worlds <n>       determinizations per decision (default 8)
  --iterations <n>   search iterations per world (default 400; higher is
                     stronger and slower)
  --first            you go first (default: decided by the seed)
  --log <path>       where to write the game log (default game.log)

At a prompt, type the number of the move you want. `?` reprints the position,
`q` resigns and still writes the log.";

struct Args {
    seed: u64,
    net_path: String,
    heuristic: bool,
    worlds: u32,
    iterations: u32,
    human_first: Option<bool>,
    log_path: String,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        std::process::exit(0);
    }
    const KNOWN: &[&str] = &[
        "--seed",
        "--net",
        "--heuristic",
        "--worlds",
        "--iterations",
        "--first",
        "--log",
    ];
    for a in raw.iter() {
        if a.starts_with('-') && !KNOWN.contains(&a.as_str()) {
            eprintln!("unknown option {a}\n\n{USAGE}");
            std::process::exit(2);
        }
    }
    let get = |flag: &str| -> Option<String> {
        raw.iter()
            .position(|a| a == flag)
            .and_then(|i| raw.get(i + 1))
            .cloned()
    };
    Args {
        seed: get("--seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(1)
            }),
        net_path: get("--net").unwrap_or_else(|| "models/net.bin".into()),
        heuristic: raw.iter().any(|a| a == "--heuristic"),
        worlds: get("--worlds").and_then(|s| s.parse().ok()).unwrap_or(8),
        iterations: get("--iterations")
            .and_then(|s| s.parse().ok())
            .unwrap_or(400),
        human_first: raw.iter().any(|a| a == "--first").then_some(true),
        log_path: get("--log").unwrap_or_else(|| "game.log".into()),
    }
}

/// Cards in a pile, grouped and counted, so a 40-card deck does not print as
/// 40 words.
fn tally(cards: &[Card]) -> String {
    let mut counts: Vec<(Card, usize)> = Vec::new();
    for c in cards {
        match counts.iter_mut().find(|(x, _)| x == c) {
            Some((_, n)) => *n += 1,
            None => counts.push((*c, 1)),
        }
    }
    counts.sort_by_key(|(c, _)| (c.cost(), format!("{c}")));
    counts
        .iter()
        .map(|(c, n)| {
            if *n == 1 {
                format!("{c}")
            } else {
                format!("{c} x{n}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn show_position(state: &GameState, human: usize, d: &Decision) {
    let me = &state.players[human];
    let them = &state.players[1 - human];

    println!("\n{}", "-".repeat(66));

    // Supply, with the piles that decide the ending called out.
    let mut line: Vec<String> = Vec::new();
    let mut empty = 0;
    for i in 0..dominion_core::NUM_CARDS {
        if !state.in_supply[i] {
            continue;
        }
        let card = Card::from_idx(i);
        let n = state.supply[i];
        if n == 0 {
            empty += 1;
        }
        line.push(format!("{card}(${}) {n}", card.cost()));
    }
    println!("supply: {}", line.join("  "));
    if empty > 0 {
        println!("        {empty} pile(s) empty — 3 ends the game");
    }

    println!(
        "\nyou:  {} cards, {} VP   |   AI: {} cards, {} VP",
        me.total_cards(),
        me.score(),
        them.total_cards(),
        them.score()
    );
    // Every card either player owns is public in Dominion — gains, trashes and
    // discards all happen face up — so showing both decks is the real
    // information a player has, not a concession.
    println!(
        "all your cards: {}",
        tally(&me.all_cards().collect::<Vec<_>>())
    );
    println!(
        "all AI cards:   {}",
        tally(&them.all_cards().collect::<Vec<_>>())
    );

    if !me.play.is_empty() {
        println!("in play:   {}", tally(&me.play));
    }
    println!("your hand: {}", tally(&me.hand));
    println!(
        "actions {}  buys {}  coins ${}",
        me.actions, me.buys, me.coins
    );
    println!("{}", prompt_for(d.ctx));
}

fn prompt_for(ctx: Ctx) -> String {
    let what = match ctx {
        Ctx::ActionPhase => "play an Action, or end the phase",
        Ctx::BuyPhase => "play a Treasure, buy a card, or end your turn",
        Ctx::MoatReveal => "reveal Moat to block the attack?",
        Ctx::CellarDiscard => "Cellar: discard a card (then draw that many)",
        Ctx::ChapelTrash => "Chapel: trash up to 4 cards",
        Ctx::HarbingerTopdeck => "Harbinger: put a card from your discard on top",
        Ctx::VassalPlay => "Vassal: play the discarded Action?",
        Ctx::WorkshopGain => "Workshop: gain a card costing up to $4",
        Ctx::BureaucratReveal => "Bureaucrat: reveal a Victory card to topdeck",
        Ctx::MilitiaDiscard => "Militia: discard down to 3 cards",
        Ctx::MoneylenderTrash => "Moneylender: trash a Copper for +$3?",
        Ctx::PoacherDiscard => "Poacher: discard (one per empty pile)",
        Ctx::RemodelTrash => "Remodel: trash a card",
        Ctx::RemodelGain => "Remodel: gain a card costing up to $2 more",
        Ctx::ThroneRoomPlay => "Throne Room: choose an Action to play twice",
        Ctx::BanditTrash => "Bandit: choose which revealed Treasure is trashed",
        Ctx::LibrarySetAside => "Library: set this Action aside instead of keeping it?",
        Ctx::MineTrash => "Mine: trash a Treasure",
        Ctx::MineGain => "Mine: gain a Treasure costing up to $3 more",
        Ctx::SentryTrash => "Sentry: trash any of the revealed cards",
        Ctx::SentryDiscard => "Sentry: discard any of the rest",
        Ctx::SentryOrder => "Sentry: choose which card goes on top",
        Ctx::ArtisanGain => "Artisan: gain a card costing up to $5",
        Ctx::ArtisanTopdeck => "Artisan: put a card from hand on your deck",
    };
    format!("\n>> {what}")
}

/// Read a choice, re-prompting until it is one. Returns `None` if the player
/// resigns or stdin closes.
fn ask(options: &[Move], state: &GameState, human: usize, d: &Decision) -> Option<Move> {
    let stdin = std::io::stdin();
    loop {
        for (i, mv) in options.iter().enumerate() {
            let extra = match mv {
                Move::Buy(c) => format!("  (${})", c.cost()),
                _ => String::new(),
            };
            println!("  {i}) {mv}{extra}");
        }
        print!("choice: ");
        std::io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).ok()? == 0 {
            return None; // EOF
        }
        let line = line.trim();
        match line {
            "q" | "quit" | "resign" => return None,
            "?" => {
                show_position(state, human, d);
                continue;
            }
            _ => {}
        }
        match line.parse::<usize>() {
            Ok(i) if i < options.len() => return Some(options[i]),
            _ => println!(
                "  — type a number from 0 to {}, or ? / q",
                options.len() - 1
            ),
        }
    }
}

fn main() {
    let args = parse_args();

    let mut krng = Rng::new(args.seed);
    let kingdom = Game::random_kingdom(&mut krng);
    let mut game = Game::new(&kingdom, 2, args.seed).expect("new game");

    // Whoever the engine puts on move first is player 0; --first claims it.
    let human = if args.human_first.unwrap_or(args.seed % 2 == 0) {
        0
    } else {
        1
    };

    let cfg = MctsConfig {
        worlds: args.worlds,
        iterations: args.iterations,
        ..Default::default()
    };

    let net = if args.heuristic {
        None
    } else {
        match Net::load(&args.net_path) {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!(
                    "could not load {}: {e}\nfalling back to the heuristic-guided search \
                     (pass --heuristic to silence this)",
                    args.net_path
                );
                None
            }
        }
    };
    let mut ai: Box<dyn Agent> = match &net {
        Some(n) => Box::new(NetMctsAgent::new(cfg, n)),
        None => Box::new(MctsAgent::new(cfg)),
    };

    let mut log = GameLog::new(kingdom.clone(), 2, args.seed);

    println!("Dominion — Base 2E.  seed {}", args.seed);
    println!("you are player {}, the AI is player {}", human, 1 - human);
    println!("opponent: {}", ai.name());
    println!("kingdom: {}", tally(&kingdom));

    let mut resigned = false;
    while !game.is_over() {
        let d = game.decision().expect("live game has a decision").clone();
        let options = d.options.clone();

        let mv = if d.player == human {
            show_position(&game.state, human, &d);
            match ask(&options, &game.state, human, &d) {
                Some(mv) => mv,
                None => {
                    resigned = true;
                    break;
                }
            }
        } else {
            let mv = ai.decide(&game.state, &d);
            // Only narrate what a real opponent's move would reveal anyway.
            if matches!(d.ctx, Ctx::ActionPhase | Ctx::BuyPhase) || options.len() > 1 {
                println!("AI: {mv}");
            }
            mv
        };

        game.apply(mv).expect("chosen move is legal");
        log.moves.push(mv);
    }

    if let Err(e) = std::fs::write(&args.log_path, log.to_text()) {
        eprintln!("could not write {}: {e}", args.log_path);
    } else {
        println!(
            "\ngame log written to {} — `advise {} --ply N` replays any position",
            args.log_path, args.log_path
        );
    }

    if resigned {
        println!("resigned.");
        return;
    }

    let scores = game.state.scores();
    println!("\n{}", "=".repeat(66));
    println!("final: you {} — AI {}", scores[human], scores[1 - human]);
    let results = game.state.results();
    println!(
        "{}",
        match results[human] {
            r if r > 0.75 => "you win.",
            r if r < 0.25 => "the AI wins.",
            _ => "a tie.",
        }
    );
}
