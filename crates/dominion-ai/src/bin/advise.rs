//! Ask the AI what to play at a given point in a game.
//!
//! ```text
//! cargo run --release --bin advise -- game.log          # advise on the end
//! cargo run --release --bin advise -- game.log --ply 42 # advise partway in
//! cargo run --release --bin advise -- --demo            # generate and advise
//! ```
//!
//! The log format is the one [`dominion_core::GameLog`] writes: a kingdom, a
//! seed, and the moves so far. Because the engine is deterministic given those,
//! the position is rebuilt exactly — no need to record any state.

use dominion_ai::{advise_log, MctsConfig};
use dominion_bots::policy::HeuristicBot;
use dominion_bots::Agent;
use dominion_core::{Game, GameLog, RecordedGame, Rng};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = MctsConfig {
        worlds: 12,
        iterations: 800,
        ..Default::default()
    };
    let mut rng = Rng::new(0xA11CE);

    if args.is_empty() || args[0] == "--demo" {
        demo(&cfg, &mut rng);
        return;
    }

    let path = &args[0];
    let ply = args
        .iter()
        .position(|a| a == "--ply")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok());

    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(1);
        }
    };
    let log = match GameLog::from_text(&text) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    };

    match advise_log(&log, ply, &cfg, &mut rng) {
        Ok(advice) => print!("{advice}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

/// Play a few turns of a real game, then advise on the position reached — so
/// the feature can be seen working without having a log to hand.
fn demo(cfg: &MctsConfig, rng: &mut Rng) {
    let kingdom = Game::random_kingdom(&mut Rng::new(31));
    let mut rec = RecordedGame::new(&kingdom, 2, 2024).unwrap();
    let mut bot = HeuristicBot;

    // Stop at a genuine buy choice: turn 5 or later, treasures already played
    // (otherwise playing one is forced and there is nothing to advise on), and
    // something actually affordable.
    while !rec.game.is_over() {
        let d = rec.game.decision().unwrap().clone();
        let treasures_left = d
            .options
            .iter()
            .any(|m| matches!(m, dominion_core::Move::Play(c) if c.is_treasure()));
        let can_buy = d
            .options
            .iter()
            .filter(|m| matches!(m, dominion_core::Move::Buy(_)))
            .count()
            > 1;
        if d.ctx == dominion_core::Ctx::BuyPhase
            && rec.game.state.players[d.player].turns >= 5
            && !treasures_left
            && can_buy
        {
            break;
        }
        let mv = bot.decide(&rec.game.state, &d);
        rec.apply(mv).unwrap();
    }

    let kingdom_names: Vec<&str> = kingdom.iter().map(|c| c.name()).collect();
    println!("kingdom: {}", kingdom_names.join(", "));
    println!("log is {} moves long\n", rec.log.moves.len());

    match advise_log(&rec.log, None, cfg, rng) {
        Ok(advice) => print!("{advice}"),
        Err(e) => eprintln!("{e}"),
    }

    println!("\n--- the log itself ---");
    let text = rec.log.to_text();
    let mut lines = text.lines();
    for line in lines.by_ref().take(5) {
        println!("{line}");
    }
    println!(
        "  ... ({} more moves)",
        rec.log.moves.len().saturating_sub(1)
    );
}
