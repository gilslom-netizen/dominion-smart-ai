//! Is the search actually deciding anything, or just returning its prior?
//!
//! Sixteen times the search budget bought nothing (50.42% +/- 4.56 against the
//! same network at 1x). In a healthy MCTS that would be four doublings and
//! worth hundreds of Elo, so the search is not converging on better moves with
//! more samples — it is converging on the same move.
//!
//! The obvious suspect is the prior. PUCT explores in proportion to
//! `c * P * sqrt(N) / (1 + n)`, so a prior concentrated hard enough on one move
//! makes every extra iteration reinforce that move instead of testing the
//! alternatives. If that is what is happening, the search is an expensive
//! identity function over its prior, and every result in this project is
//! really a result about the prior.
//!
//! This measures it directly: how often does the search's pick differ from the
//! prior's top move, and how much of the visit mass does the prior's favourite
//! take, at each budget.

use dominion_ai::evaluator::{Evaluator, NetEvaluator};
use dominion_ai::{mcts, prior, MctsConfig, Net};
use dominion_core::{Ctx, Game, Rng};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().cloned().unwrap_or_else(|| "models/net.bin".into());
    let games: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let net = Net::load(&path).expect("load network");
    let eval = NetEvaluator::new(&net);

    println!("{:>10}  {:>10}  {:>14}  {:>16}", "budget", "decisions", "search != prior", "top-move visit share");

    for (w, i) in [(1u32, 50u32), (4, 200), (8, 400), (16, 800)] {
        let cfg = MctsConfig { worlds: w, iterations: i, ..Default::default() };
        let mut rng = Rng::new(4242);
        let (mut decisions, mut disagreements, mut share_sum) = (0u32, 0u32, 0.0f64);

        for g in 0..games {
            let kingdom = Game::random_kingdom(&mut Rng::new(g as u64 + 1));
            let mut game = Game::new(&kingdom, 2, g as u64 * 7 + 3).unwrap();
            let mut steps = 0;
            while !game.is_over() && steps < 60 {
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
                    let top_visits = out
                        .visits
                        .iter()
                        .find(|(m, _)| *m == top)
                        .map(|(_, v)| *v)
                        .unwrap_or(0);

                    decisions += 1;
                    if out.best != top {
                        disagreements += 1;
                    }
                    share_sum += top_visits as f64 / total as f64;
                    game.apply(out.best).unwrap();
                } else {
                    game.apply(options[0]).unwrap();
                }
                steps += 1;
            }
        }
        println!(
            "{:>10}  {:>10}  {:>13.1}%  {:>15.1}%",
            format!("{w}x{i}"),
            decisions,
            100.0 * disagreements as f64 / decisions.max(1) as f64,
            100.0 * share_sum / decisions.max(1) as f64
        );
    }
    println!("\nIf disagreement stays near zero as the budget grows, the search is");
    println!("reproducing its prior and the extra compute is doing nothing.");
}
