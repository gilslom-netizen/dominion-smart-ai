//! Generate a shard of self-play training data.
//!
//! Meant to run on more than one machine at once: give each machine a distinct
//! `--tag` and they will never write the same file, so there is nothing to
//! coordinate before `train` reads everything back.
//!
//! ```text
//! cargo run --release --bin selfplay -- --games 500 --threads 12 --tag laptop
//! cargo run --release --bin selfplay -- --games 200 --threads 4  --tag cloud --net models/net.bin
//! ```
//!
//! Without `--net`, generation uses the heuristic-guided search already
//! measured in the README (60.4% vs the heuristic). With `--net`, it uses that
//! network's policy and value instead — the loop that is supposed to make each
//! generation stronger than the last.
//!
//! Output is the compact `.gamelog` format: the game itself rather than the
//! expanded feature vectors, roughly 20x smaller, and small enough to commit so
//! that several machines' self-play can actually be pooled into one training
//! run. See [`dominion_ai::compact`].
//!
//! **Completed games are written to the file as they finish**, not buffered
//! until the end. A run of a few thousand games takes hours, and holding it all
//! in memory meant any interruption — a crash, a reboot, or simply wanting to
//! change the code — threw the whole thing away. With incremental writes the
//! shard on disk is valid and usable at every moment, so stopping a run costs
//! at most the handful of games in flight. Restarting after a code change needs
//! no coordination either: the new run writes its own shard, and `train` reads
//! every shard it finds.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use dominion_ai::compact;
use dominion_ai::evaluator::{HeuristicEvaluator, NetEvaluator, RolloutEvaluator};
use dominion_ai::selfplay::{play_selfplay_game_recorded, DEFAULT_LAMBDA};
use dominion_ai::{MctsConfig, Net};
use dominion_core::{Game, Rng};

struct Args {
    games: u32,
    threads: usize,
    tag: String,
    net_path: Option<String>,
    worlds: u32,
    iterations: u32,
    seed: u64,
    lambda: f32,
    /// Price search leaves with a heuristic rollout instead of the network's
    /// value head. Measured much better calibrated (Brier 0.1301 vs 0.1599)
    /// and worth +101 Elo game-matched, at 4.2x the cost per game.
    rollout_leaves: bool,
}

const USAGE: &str = "\
generate self-play training data

usage: selfplay [options]

  --tag <name>       names the output file; also seeds generation unless
                     --seed is given, so two tags never produce the same games
  --games <n>        how many to play (default 200)
  --threads <n>      default: all cores
  --net <path>       guide the search with a trained network
  --worlds <n>       determinizations per decision (default 8)
  --iterations <n>   search iterations per world (default 300)
  --seed <n>         override the tag-derived seed
  --lambda <f>       TD(lambda) weight (default 0.9)
  --rollout-leaves   price leaves by rollout, not the network's value head.
                     Better targets (+101 Elo at equal search) but ~4.2x
                     slower per game, so expect roughly a quarter the games
                     in the same wall clock.

Output goes to selfplay-data/<tag>-<timestamp>.gamelog and is flushed after
every game, so the run can be stopped at any time without losing work.";

/// Derive a seed from the tag.
///
/// The default used to be a literal 0, which meant every machine that did not
/// pass --seed generated *byte-identical* games — same kingdoms, same shuffles,
/// same everything. Several machines pooling that data would have been pooling
/// the same games several times over, and nothing in the output would have
/// hinted at it. Seeding from the tag keeps runs reproducible (same tag, same
/// data) while making collisions between differently-named runs impossible.
fn seed_from_tag(tag: &str) -> u64 {
    // FNV-1a: tiny, no dependency, and good enough to separate short names.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in tag.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        std::process::exit(0);
    }
    let get = |flag: &str| -> Option<String> {
        raw.iter()
            .position(|a| a == flag)
            .and_then(|i| raw.get(i + 1))
            .cloned()
    };
    // An unknown flag almost always means a typo, and silently ignoring it
    // starts a multi-hour run with settings the caller did not ask for.
    const KNOWN: &[&str] = &[
        "--tag",
        "--games",
        "--threads",
        "--net",
        "--worlds",
        "--iterations",
        "--seed",
        "--lambda",
        "--rollout-leaves",
    ];
    for (i, a) in raw.iter().enumerate() {
        if a.starts_with('-') && !KNOWN.contains(&a.as_str()) {
            eprintln!("unknown option {a}\n\n{USAGE}");
            std::process::exit(2);
        }
        // Skip the value that follows a known flag, so a value like "-1"
        // is not mistaken for an option.
        if a.starts_with("--") && KNOWN.contains(&a.as_str()) && i + 1 < raw.len() {
            continue;
        }
    }
    let tag = get("--tag").unwrap_or_else(|| "default".into());
    Args {
        games: get("--games").and_then(|s| s.parse().ok()).unwrap_or(200),
        threads: get("--threads")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            }),
        net_path: get("--net"),
        worlds: get("--worlds").and_then(|s| s.parse().ok()).unwrap_or(8),
        iterations: get("--iterations")
            .and_then(|s| s.parse().ok())
            .unwrap_or(300),
        seed: get("--seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| seed_from_tag(&tag)),
        tag,
        lambda: get("--lambda")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_LAMBDA),
        rollout_leaves: raw.iter().any(|a| a == "--rollout-leaves"),
    }
}

fn main() {
    let args = parse_args();
    std::fs::create_dir_all("selfplay-data").ok();
    let out_path = format!(
        "selfplay-data/{}-{}.gamelog",
        args.tag,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );

    let net = args.net_path.as_ref().map(|p| {
        Net::load(p).unwrap_or_else(|e| {
            eprintln!("cannot load {p}: {e}");
            std::process::exit(1);
        })
    });

    let cfg = MctsConfig {
        worlds: args.worlds,
        iterations: args.iterations,
        ..Default::default()
    };

    println!(
        "generating {} games on {} threads -> {out_path}{}",
        args.games,
        args.threads,
        if net.is_some() {
            " (net-guided)"
        } else {
            " (heuristic-guided)"
        }
    );
    println!("TD(lambda) = {}", args.lambda);
    println!("games are flushed as they finish; this run can be stopped at any time");

    let done = AtomicU32::new(0);
    // Threads used to take a fixed slice of the game range. That stalls the run
    // whenever the slices differ in cost, and it degrades badly when the games
    // do not divide evenly by threads: 13 games on 12 threads gives one thread
    // two games and the rest one, doubling wall time. They now claim the next
    // index from a shared counter.
    //
    // This was *not* measured as a throughput win: on a 12-thread 15W laptop
    // parallel efficiency came out ~6.7 of 12 cores busy either way, because
    // there the limit is the package power budget, not an idle tail. (Measured
    // against the previous PIMC search, so the numbers predate ISMCTS; the
    // power ceiling is a property of the machine, not of the search.) The
    // concrete win is reproducibility: every RNG below is derived from the game
    // index rather than the thread number, so a given `--seed` reproduces the
    // same shard regardless of thread count or scheduling — not true before.
    let next = AtomicU32::new(0);
    let games = args.games;
    let start = Instant::now();

    // Serializes appends so two threads never interleave a record, and so the
    // magic header is written exactly once.
    let writer = Mutex::new(());
    let written = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        let net = &net;
        let done = &done;
        let next = &next;
        let writer = &writer;
        let written = &written;
        let out_path = &out_path;
        let handles: Vec<_> = (0..args.threads)
            .map(|_| {
                scope.spawn(move || {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= games {
                            break;
                        }
                        let game_seed = args
                            .seed
                            .wrapping_mul(0x2545F4914F6CDD1D)
                            .wrapping_add(i as u64);
                        let mut rng = Rng::new(game_seed ^ 0x9E3779B97F4A7C15);
                        let mut krng = Rng::new(game_seed ^ 0xABCD);
                        let kingdom = Game::random_kingdom(&mut krng);
                        let record = match (&net, args.rollout_leaves) {
                            (Some(n), false) => {
                                let eval = NetEvaluator::new(n);
                                play_selfplay_game_recorded(
                                    &kingdom,
                                    2,
                                    game_seed,
                                    &cfg,
                                    &eval,
                                    &mut rng,
                                    args.lambda,
                                )
                            }
                            (Some(n), true) => {
                                let eval = RolloutEvaluator { net: n };
                                play_selfplay_game_recorded(
                                    &kingdom,
                                    2,
                                    game_seed,
                                    &cfg,
                                    &eval,
                                    &mut rng,
                                    args.lambda,
                                )
                            }
                            (None, _) => play_selfplay_game_recorded(
                                &kingdom,
                                2,
                                game_seed,
                                &cfg,
                                &HeuristicEvaluator,
                                &mut rng,
                                args.lambda,
                            ),
                        };
                        // Flush this game before touching the next one, so an
                        // interrupted run keeps everything already finished.
                        {
                            let _guard = writer.lock().unwrap_or_else(|e| e.into_inner());
                            if let Err(e) = compact::append_games(out_path, &[record.clone()]) {
                                eprintln!("failed to append to {out_path}: {e}");
                                std::process::exit(1);
                            }
                        }
                        written.fetch_add(record.decisions.len(), Ordering::Relaxed);

                        let n_done = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n_done % 10 == 0 || n_done == games {
                            let secs = start.elapsed().as_secs_f64();
                            eprintln!(
                                "{n_done}/{games} games  ({:.2} games/s)",
                                n_done as f64 / secs.max(1e-9)
                            );
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    });

    let size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "wrote {} decisions from {} games to {out_path} in {:.1}s ({:.1} MB)",
        written.load(Ordering::Relaxed),
        done.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64(),
        size as f64 / 1e6
    );
    println!(
        "this file is small enough to commit — push it so other machines can train on it too."
    );
}

#[cfg(test)]
mod tests {
    use super::seed_from_tag;

    /// Two machines that both omit --seed must not generate the same games.
    /// The default used to be a literal 0, so every agent produced a
    /// byte-identical corpus and pooling their work added nothing.
    #[test]
    fn different_tags_give_different_seeds() {
        let tags = [
            "alice", "bob", "agent-a", "agent-b", "ccweb", "cwopus", "xdtvd7", "opus5a", "cloud",
            "laptop", "aws", "default", "w1", "w2",
        ];
        let seeds: Vec<u64> = tags.iter().map(|t| seed_from_tag(t)).collect();
        for (i, a) in seeds.iter().enumerate() {
            for (j, b) in seeds.iter().enumerate().skip(i + 1) {
                assert_ne!(a, b, "{} and {} collide", tags[i], tags[j]);
            }
        }
    }

    /// Names differing by one character must still separate — agents pick
    /// things like w1/w2, and a weak hash would map those close together.
    #[test]
    fn near_identical_tags_separate() {
        for (a, b) in [("w1", "w2"), ("agent1", "agent2"), ("a", "b"), ("x", "xx")] {
            let (sa, sb) = (seed_from_tag(a), seed_from_tag(b));
            assert_ne!(sa, sb);
            // and not merely adjacent, which would give overlapping game
            // sequences once the seed is multiplied out per game index
            assert!(
                sa.abs_diff(sb) > 1000,
                "{a} and {b} seed too close: {sa} vs {sb}"
            );
        }
    }

    #[test]
    fn the_same_tag_stays_reproducible() {
        assert_eq!(seed_from_tag("cloud"), seed_from_tag("cloud"));
    }
}
