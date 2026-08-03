//! How fast is one training step?
//!
//! Training is now the slowest part of an iteration — a 512x256 network takes
//! roughly three quarters of an hour for six epochs — so it is worth being able
//! to measure it directly rather than inferring it from whole-run wall clock.
//!
//! Prints steps/second and the implied time for a full run at both network
//! widths in use.

use std::time::Instant;

use dominion_ai::features::FEATURE_DIM;
use dominion_ai::Net;
use dominion_core::Rng;

fn bench(hidden1: usize, hidden2: usize, steps: usize) -> f64 {
    let mut rng = Rng::new(1);
    let mut net = Net::with_hidden(hidden1, hidden2, &mut rng);

    let mut x = [0.0f32; FEATURE_DIM];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 7) as f32 - 3.0) / 10.0;
    }
    let legal: Vec<usize> = (0..8).collect();
    let targets: Vec<f32> = vec![0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05];

    // Warm the caches so the first iterations do not skew a short run.
    for _ in 0..200 {
        net.train_step(&x, &legal, &targets, 0.6, 0.01);
    }

    let t = Instant::now();
    for _ in 0..steps {
        net.train_step(&x, &legal, &targets, 0.6, 0.01);
    }
    steps as f64 / t.elapsed().as_secs_f64()
}

fn main() {
    // 6 epochs over the corpus this project has been training on.
    let full_run = 736_727u64 * 6;

    println!(
        "{:>12}  {:>14}  {:>18}",
        "network", "steps/s", "6 epochs over 737k"
    );
    for (h1, h2) in [(128usize, 64usize), (512, 256)] {
        let rate = bench(h1, h2, 40_000);
        println!(
            "{:>12}  {:>14.0}  {:>15.1} min",
            format!("{h1}x{h2}"),
            rate,
            full_run as f64 / rate / 60.0
        );
    }
    println!("\nNote: run this on an idle machine. Self-play generation saturates");
    println!("every core, and timing against a busy box measures the contention.");
}
