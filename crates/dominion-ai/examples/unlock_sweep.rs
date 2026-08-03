//! Can the search be made to actually use its budget?
//!
//! Measured: 16x the search budget changed the chosen move in 5.7% of
//! decisions and won 50.42% +/- 4.56 against 1x. The search reproduces its
//! prior. This sweeps the two knobs that control how far it is allowed to
//! disagree — prior temperature (flatten the network's policy) and the PUCT
//! exploration constant — and reports, for each setting, how often 16x
//! actually decides differently from 1x.
//!
//! Disagreement alone is not the goal: a search that ignores its prior
//! entirely disagrees constantly and plays badly, which is exactly how this
//! project's first UCT attempt lost every game. So the sweep reports
//! disagreement as the diagnostic and leaves strength to a head-to-head at
//! whichever settings look promising.

use dominion_ai::evaluator::{Evaluator, NetEvaluator};
use dominion_ai::{mcts, prior, MctsConfig, Net};
use dominion_core::{Ctx, Game, Rng};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "models/net.bin".into());
    let games: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let net = Net::load(&path).expect("load network");

    println!(
        "{:>6} {:>6}  {:>10}  {:>16}  {:>18}",
        "temp", "c_puct", "decisions", "16x != prior top", "top-move visit share"
    );

    for &temp in &[1.0f32, 1.5, 2.5, 4.0] {
        for &c in &[2.5f32, 6.0, 12.0] {
            let eval = NetEvaluator::with_temperature(&net, temp);
            let cfg = MctsConfig {
                worlds: 16,
                iterations: 800,
                exploration: c,
                ..Default::default()
            };
            let mut rng = Rng::new(31337);
            let (mut n, mut diff, mut share) = (0u32, 0u32, 0.0f64);

            for g in 0..games {
                let kingdom = Game::random_kingdom(&mut Rng::new(g as u64 + 1));
                let mut game = Game::new(&kingdom, 2, g as u64 * 7 + 3).unwrap();
                let mut steps = 0;
                while !game.is_over() && steps < 40 {
                    let d = game.decision().unwrap().clone();
                    let options = prior::restrict(&game.state, &d);
                    if options.len() > 1 && d.ctx == Ctx::BuyPhase {
                        let priors = eval.priors(&game.state, &d, &options);
                        let top = options
                            .iter()
                            .zip(&priors)
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(m, _)| *m)
                            .unwrap();
                        let out = mcts::search_full(&game.state, &d, &cfg, &eval, &mut rng);
                        let total: u32 = out.visits.iter().map(|(_, v)| v).sum::<u32>().max(1);
                        let tv = out
                            .visits
                            .iter()
                            .find(|(m, _)| *m == top)
                            .map(|(_, v)| *v)
                            .unwrap_or(0);
                        n += 1;
                        if out.best != top {
                            diff += 1;
                        }
                        share += tv as f64 / total as f64;
                        game.apply(out.best).unwrap();
                    } else {
                        game.apply(options[0]).unwrap();
                    }
                    steps += 1;
                }
            }
            println!(
                "{:>6.1} {:>6.1}  {:>10}  {:>15.1}%  {:>17.1}%",
                temp,
                c,
                n,
                100.0 * diff as f64 / n.max(1) as f64,
                100.0 * share / n.max(1) as f64
            );
        }
    }
    println!("\nBaseline was temp 1.0 / c 2.5: 5.7% disagreement, 71% visit share.");
}
