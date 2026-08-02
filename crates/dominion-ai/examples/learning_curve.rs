//! How much is more self-play data actually worth?
//!
//! ```text
//! cargo run --release --example learning_curve -- [eval_pairs]
//! ```
//!
//! Trains the same architecture on growing prefixes of the available data and
//! plays each result against the one before it. That last part matters: the
//! question is not "how good is a network trained on N games" but "does going
//! from N to 2N buy anything", and a direct match answers it with far less
//! noise than comparing two win rates against a common opponent.
//!
//! This exists to make a spending decision measurable. Buying a machine that
//! generates ten times more self-play is only worth it while the curve is still
//! climbing; if it has flattened, the money buys nothing and the bottleneck is
//! elsewhere.

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{compact, example, Example, MctsConfig, Net};
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::Agent;
use dominion_core::Rng;

fn load_all() -> Vec<Example> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir("selfplay-data") else {
        return out;
    };
    let mut paths: Vec<std::path::PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    for p in paths {
        let name = p.to_string_lossy().into_owned();
        match p.extension().and_then(|x| x.to_str()) {
            Some("gamelog") => out.extend(compact::read_examples(&name).unwrap_or_default()),
            Some("shard") => out.extend(example::read_shard(&name).unwrap_or_default()),
            _ => {}
        }
    }
    out
}

fn train_on(examples: &[Example], epochs: u32, rng: &mut Rng) -> Net {
    let mut net = Net::new(rng);
    let mut order: Vec<usize> = (0..examples.len()).collect();
    for _ in 0..epochs {
        rng.shuffle(&mut order);
        for &i in &order {
            let ex = &examples[i];
            let idx: Vec<usize> = ex.policy.iter().map(|(m, _)| m.index()).collect();
            let tgt: Vec<f32> = ex.policy.iter().map(|(_, p)| *p).collect();
            net.train_step(&ex.features, &idx, &tgt, ex.td_target, 0.01);
        }
    }
    net
}

fn main() {
    let pairs: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let mut rng = Rng::new(0xC0DE);
    let mut all = load_all();
    if all.is_empty() {
        eprintln!("nothing in selfplay-data/");
        std::process::exit(1);
    }
    rng.shuffle(&mut all);
    println!("{} examples available\n", all.len());

    // Quarter, half, then everything.
    let sizes = [all.len() / 4, all.len() / 2, all.len()];
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let cfg = MctsConfig {
        worlds: 4,
        iterations: 200,
        ..Default::default()
    };

    let mut previous: Option<(usize, Net)> = None;
    for &n in &sizes {
        let net = train_on(&all[..n], 6, &mut rng);
        match &previous {
            None => println!("trained on {n} examples (baseline)"),
            Some((prev_n, prev_net)) => {
                let (a, b) = (&net, prev_net);
                let res = run_match_parallel(
                    move || Box::new(NetMctsAgent::new(cfg, a)) as Box<dyn Agent>,
                    move || Box::new(NetMctsAgent::new(cfg, b)) as Box<dyn Agent>,
                    pairs,
                    0x5115,
                    &Kingdoms::Random,
                    cores,
                );
                let sigma = if res.stderr() > 0.0 {
                    (res.win_rate_a() - 0.5).abs() / res.stderr()
                } else {
                    0.0
                };
                println!(
                    "{n} examples vs {prev_n}: {:.2}% ± {:.2}  ({:.1} sigma)",
                    res.win_rate_a() * 100.0,
                    res.stderr() * 100.0,
                    sigma
                );
            }
        }
        previous = Some((n, net));
    }

    println!("\nA gap that shrinks toward even as the data doubles means the curve is");
    println!("flattening, and more self-play machines buy progressively less.");
}
