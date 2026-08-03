//! Is the search's value estimate better than the network's, or worse?
//!
//! `unlock_strength` measured that flattening the prior makes play worse, and
//! worse in proportion to the search budget: -83 Elo at 8x400, -168 Elo at
//! 16x800. Extra simulations actively destroy strength. The only way that
//! happens is if what the tree backs up is worse than the prior it overrides,
//! so this measures that quantity directly instead of inferring it from game
//! results.
//!
//! For each searched decision it records three numbers about the player to
//! move: the network's raw value head, the search's root value at several
//! budgets, and — after playing the game out — what actually happened. Brier
//! score (mean squared error against the 0/1 outcome) is the headline; a
//! search that is doing its job should score *below* the raw network, and
//! should improve as the budget grows.
//!
//! The play itself uses the locked prior (temperature 1.0), i.e. the
//! configuration the system is actually strongest in, so the positions are
//! drawn from the distribution that matters rather than from a setting
//! already known to be bad.

use dominion_ai::evaluator::{Evaluator, NetEvaluator};
use dominion_ai::{mcts, prior, MctsConfig, Net};
use dominion_core::{Ctx, Game, Rng};

/// Budgets to price each position at, as (worlds, iterations).
const BUDGETS: [(u32, u32); 3] = [(4, 200), (8, 400), (16, 800)];

struct Acc {
    n: u32,
    sq: f64,
    sum_pred: f64,
    sum_out: f64,
    sum_cross: f64,
    sum_pred_sq: f64,
    sum_out_sq: f64,
}

impl Acc {
    fn new() -> Self {
        Acc { n: 0, sq: 0.0, sum_pred: 0.0, sum_out: 0.0, sum_cross: 0.0, sum_pred_sq: 0.0, sum_out_sq: 0.0 }
    }
    fn push(&mut self, pred: f32, out: f32) {
        let (p, o) = (pred as f64, out as f64);
        self.n += 1;
        self.sq += (p - o) * (p - o);
        self.sum_pred += p;
        self.sum_out += o;
        self.sum_cross += p * o;
        self.sum_pred_sq += p * p;
        self.sum_out_sq += o * o;
    }
    fn brier(&self) -> f64 {
        self.sq / self.n.max(1) as f64
    }
    /// Pearson correlation between prediction and outcome.
    fn corr(&self) -> f64 {
        let n = self.n.max(1) as f64;
        let cov = self.sum_cross / n - (self.sum_pred / n) * (self.sum_out / n);
        let vp = (self.sum_pred_sq / n - (self.sum_pred / n).powi(2)).max(0.0).sqrt();
        let vo = (self.sum_out_sq / n - (self.sum_out / n).powi(2)).max(0.0).sqrt();
        if vp * vo < 1e-12 {
            0.0
        } else {
            cov / (vp * vo)
        }
    }
    fn mean_pred(&self) -> f64 {
        self.sum_pred / self.n.max(1) as f64
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().cloned().unwrap_or_else(|| "models/net.bin".into());
    let games: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let net = Net::load(&path).expect("load network");
    let eval = NetEvaluator::new(&net);

    let mut net_acc = Acc::new();
    let mut search_acc: Vec<Acc> = BUDGETS.iter().map(|_| Acc::new()).collect();
    // Predictions held until the game finishes and the outcome is known.
    let mut pending: Vec<(usize, f32, Vec<f32>)> = Vec::new();

    let mut rng = Rng::new(0xCA11B);
    for g in 0..games {
        let kingdom = Game::random_kingdom(&mut Rng::new(g as u64 + 1));
        let mut game = Game::new(&kingdom, 2, g as u64 * 13 + 5).unwrap();
        pending.clear();

        let play_cfg = MctsConfig {
            worlds: 8,
            iterations: 400,
            ..Default::default()
        };

        while !game.is_over() {
            let d = game.decision().unwrap().clone();
            let options = prior::restrict(&game.state, &d);
            if options.len() < 2 {
                game.apply(options[0]).unwrap();
                continue;
            }

            if d.ctx == Ctx::BuyPhase {
                let v_net = eval
                    .leaf_value(&game.state, d.player)
                    .expect("2-player value");
                let v_search: Vec<f32> = BUDGETS
                    .iter()
                    .map(|&(w, i)| {
                        let cfg = MctsConfig {
                            worlds: w,
                            iterations: i,
                            ..play_cfg
                        };
                        mcts::search_full(&game.state, &d, &cfg, &eval, &mut rng).value
                    })
                    .collect();
                pending.push((d.player, v_net, v_search));
            }

            let out = mcts::search_full(&game.state, &d, &play_cfg, &eval, &mut rng);
            game.apply(out.best).unwrap();
        }

        let results = game.state.results();
        for (player, v_net, v_search) in pending.drain(..) {
            let actual = results[player];
            net_acc.push(v_net, actual);
            for (acc, v) in search_acc.iter_mut().zip(&v_search) {
                acc.push(*v, actual);
            }
        }

        if (g + 1) % 5 == 0 {
            eprintln!("  {} games, {} positions", g + 1, net_acc.n);
        }
    }

    println!(
        "{} positions from {} games (locked prior, the configuration that plays best)\n",
        net_acc.n, games
    );
    println!(
        "{:>18}  {:>8}  {:>8}  {:>10}",
        "estimator", "brier", "corr", "mean pred"
    );
    println!(
        "{:>18}  {:>8.4}  {:>8.3}  {:>10.3}",
        "value head (raw)",
        net_acc.brier(),
        net_acc.corr(),
        net_acc.mean_pred()
    );
    for (&(w, i), acc) in BUDGETS.iter().zip(&search_acc) {
        println!(
            "{:>18}  {:>8.4}  {:>8.3}  {:>10.3}",
            format!("search {w}x{i}"),
            acc.brier(),
            acc.corr(),
            acc.mean_pred()
        );
    }
    println!(
        "\n{:>18}  {:>8.4}   (always predicting 0.5)",
        "baseline", 0.25
    );
    println!(
        "\nLower brier is better. A sound tree should beat the raw value head\n\
         and improve with budget. If it gets worse instead, the tree is\n\
         degrading the estimate it starts from, and the search — not the\n\
         network — is what needs fixing."
    );
}
