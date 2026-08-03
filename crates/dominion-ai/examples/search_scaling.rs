//! Does the system respond to more search at *play* time?
//!
//! Five hypotheses about the network have now been measured and none moved
//! playing strength: more data, more capacity, deeper search when generating,
//! and finally a better optimizer — which fit the targets measurably better
//! (policy loss 0.9046 -> 0.8726) and still played exactly even.
//!
//! That dissociation is the point. The network's job is to be a *prior*: it
//! orders which moves the search looks at first, and the search then corrects
//! it with simulations. A prior that imitates the search 22% more closely gets
//! corrected in the same way, so fitting it better buys nothing. If the network
//! is not the lever, the search is — and the cheapest way to find out is to
//! give the same network more of it.

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsConfig, Net};
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::Agent;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "models/net.bin".into());
    let pairs: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let net = Net::load(&path).expect("load network");
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let base = MctsConfig {
        worlds: 4,
        iterations: 200,
        ..Default::default()
    };
    println!("baseline: 4 worlds x 200 iterations\n");

    for (w, i) in [(8u32, 400u32), (16, 800)] {
        let bigger = MctsConfig {
            worlds: w,
            iterations: i,
            ..Default::default()
        };
        let factor = (w * i) as f64 / (base.worlds * base.iterations) as f64;
        let (a, b) = (&net, &net);
        let res = run_match_parallel(
            move || Box::new(NetMctsAgent::new(bigger, a)) as Box<dyn Agent>,
            move || Box::new(NetMctsAgent::new(base, b)) as Box<dyn Agent>,
            pairs,
            0xBEEF,
            &Kingdoms::Random,
            cores,
        );
        println!("{w}x{i} ({factor:.0}x the budget) vs baseline:");
        println!("  {res}");
        let sigma = (res.win_rate_a() - 0.5).abs() / res.stderr().max(1e-9);
        println!("  {sigma:.1} standard errors from even\n");
    }
    println!("Same network on both sides, so any gap is search depth alone.");
}
