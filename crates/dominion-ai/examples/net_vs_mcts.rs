use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsAgent, MctsConfig, Net};
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::Agent;

fn main() {
    let net = Net::load("models/net.bin").expect("load net");
    let cfg = MctsConfig::default(); // 8 worlds x 400 iterations, same as the original measurement
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let net_ref = &net;
    let res = run_match_parallel(
        move || Box::new(NetMctsAgent::new(cfg, net_ref)) as Box<dyn Agent>,
        || Box::new(MctsAgent::new(cfg)) as Box<dyn Agent>,
        60,
        0x1234,
        &Kingdoms::Random,
        cores,
    );
    println!("{res}");
}
