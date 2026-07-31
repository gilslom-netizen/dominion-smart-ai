use std::time::Instant;
use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsAgent, MctsConfig, Net};
use dominion_bots::match_runner::play_game;
use dominion_bots::policy::HeuristicBot;
use dominion_core::{Game, Rng};

fn main() {
    let net = Net::load("models/net.bin").expect("load net");
    let cfg = MctsConfig::default();
    let kingdom = Game::random_kingdom(&mut Rng::new(1));

    let t = Instant::now();
    let mut a = MctsAgent::new(cfg);
    let mut h = HeuristicBot;
    play_game(&kingdom, &mut [&mut a, &mut h], 7);
    println!("heuristic-guided MCTS(8x400): {:.2}s/game", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let mut a = NetMctsAgent::new(cfg, &net);
    let mut h = HeuristicBot;
    play_game(&kingdom, &mut [&mut a, &mut h], 7);
    println!("net-guided     NetMCTS(8x400): {:.2}s/game", t.elapsed().as_secs_f64());
}
