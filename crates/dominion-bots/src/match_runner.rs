//! Head-to-head evaluation.
//!
//! Two variance-reduction measures matter enough to be baked in rather than
//! left to the caller:
//!
//! * **Seat swapping.** The first player has a real edge in Dominion, so every
//!   pairing is played twice, once from each seat.
//! * **Paired seeds.** Both seatings use the same seed, so the two agents face
//!   the same shuffles. This turns a noisy comparison into a much tighter one
//!   for the same number of games.

use std::time::Instant;

use dominion_core::{Card, Game, Rng};

use crate::Agent;

#[derive(Clone, Debug, Default)]
pub struct MatchResult {
    pub name_a: String,
    pub name_b: String,
    pub games: u32,
    /// Sum of per-game results for A (a tie counts as half).
    pub score_a: f32,
    pub wins_a: u32,
    pub wins_b: u32,
    pub ties: u32,
    pub avg_turns: f32,
    pub elapsed_secs: f64,
}

impl MatchResult {
    /// A's win rate in `[0, 1]`, ties counted as half a win.
    pub fn win_rate_a(&self) -> f32 {
        self.score_a / self.games.max(1) as f32
    }

    /// Elo difference implied by the win rate, from A's point of view.
    pub fn elo_diff(&self) -> f32 {
        let p = self.win_rate_a().clamp(0.001, 0.999);
        -400.0 * ((1.0 / p as f64 - 1.0).log10()) as f32
    }

    /// Standard error of the win rate, for deciding whether a gap is real.
    pub fn stderr(&self) -> f32 {
        let p = self.win_rate_a();
        (p * (1.0 - p) / self.games.max(1) as f32).sqrt()
    }

    pub fn games_per_sec(&self) -> f64 {
        self.games as f64 / self.elapsed_secs.max(1e-9)
    }
}

impl std::fmt::Display for MatchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:>14} vs {:<14} {:>6.2}% ± {:.2}  ({}W/{}L/{}D over {} games, {:+.0} Elo, {:.1} turns, {:.0} games/s)",
            self.name_a,
            self.name_b,
            self.win_rate_a() * 100.0,
            self.stderr() * 100.0,
            self.wins_a,
            self.wins_b,
            self.ties,
            self.games,
            self.elo_diff(),
            self.avg_turns,
            self.games_per_sec(),
        )
    }
}

/// Play one game to completion. `agents[i]` answers for player `i`.
/// Returns each player's result in `[0, 1]`.
pub fn play_game(
    kingdom: &[Card],
    agents: &mut [&mut dyn Agent],
    seed: u64,
) -> (Vec<f32>, u32) {
    for a in agents.iter_mut() {
        a.reset();
    }
    let mut game = Game::new(kingdom, agents.len(), seed).expect("valid kingdom");
    while !game.is_over() {
        let d = game.decision().expect("live game has a decision").clone();
        let mv = agents[d.player].decide(&game.state, &d);
        game.apply(mv)
            .unwrap_or_else(|e| panic!("{} played illegally: {e}", agents[d.player].name()));
    }
    let turns = game.state.players[0].turns;
    (game.state.results(), turns)
}

/// How kingdoms are chosen for a match.
pub enum Kingdoms {
    /// The same 10 cards every game.
    Fixed(Vec<Card>),
    /// A fresh random kingdom per seed — the honest test of general strength.
    Random,
    /// Random, but with these cards always present (so both strategies are
    /// playable).
    RandomWith(Vec<Card>),
}

impl Kingdoms {
    fn pick(&self, rng: &mut Rng) -> Vec<Card> {
        match self {
            Kingdoms::Fixed(k) => k.clone(),
            Kingdoms::Random => Game::random_kingdom(rng),
            Kingdoms::RandomWith(must) => {
                let mut k: Vec<Card> = must.clone();
                k.sort_unstable();
                k.dedup();
                let mut pool: Vec<Card> = dominion_core::KINGDOM_CARDS
                    .iter()
                    .copied()
                    .filter(|c| !k.contains(c))
                    .collect();
                rng.shuffle(&mut pool);
                let need = 10usize.saturating_sub(k.len());
                k.extend(pool.into_iter().take(need));
                k.truncate(10);
                k.sort_unstable();
                k
            }
        }
    }
}

/// Play `pairs` seed-paired, seat-swapped games (so `2 * pairs` games total).
pub fn run_match(
    a: &mut dyn Agent,
    b: &mut dyn Agent,
    pairs: u32,
    seed: u64,
    kingdoms: &Kingdoms,
) -> MatchResult {
    let mut res = MatchResult {
        name_a: a.name(),
        name_b: b.name(),
        ..Default::default()
    };
    let start = Instant::now();
    let mut total_turns = 0u64;

    for i in 0..pairs {
        let game_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64);
        let mut krng = Rng::new(game_seed ^ 0xDEAD_BEEF);
        let kingdom = kingdoms.pick(&mut krng);

        // Seating 0: A first. Seating 1: B first, same shuffles.
        for swap in [false, true] {
            let (r0, turns) = if swap {
                let (r, t) = play_game(&kingdom, &mut [b, a], game_seed);
                (vec![r[1], r[0]], t)
            } else {
                play_game(&kingdom, &mut [a, b], game_seed)
            };
            total_turns += turns as u64;
            res.games += 1;
            res.score_a += r0[0];
            if r0[0] > r0[1] {
                res.wins_a += 1;
            } else if r0[0] < r0[1] {
                res.wins_b += 1;
            } else {
                res.ties += 1;
            }
        }
    }

    res.avg_turns = total_turns as f32 / res.games.max(1) as f32;
    res.elapsed_secs = start.elapsed().as_secs_f64();
    res
}
