use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsAgent, MctsConfig, Net};
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::Agent;

fn main() {
    let net = Net::load("models/net.bin").expect("load net");
    let heuristic_cfg = MctsConfig::default(); // 8x400
    let net_cfg = MctsConfig { iterations: 660, ..heuristic_cfg }; // ~equal wall-clock, per measured 1.65x speedup
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let net_ref = &net;
    let res = run_match_parallel(
        move || Box::new(NetMctsAgent::new(net_cfg, net_ref)) as Box<dyn Agent>,
        || Box::new(MctsAgent::new(heuristic_cfg)) as Box<dyn Agent>,
        60,
        0x5678,
        &Kingdoms::Random,
        cores,
    );
    println!("wall-clock-matched: {res}");
}
