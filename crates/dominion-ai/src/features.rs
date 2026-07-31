//! Turning a decision into a fixed-size vector a network can consume.
//!
//! Dominion's public information is exactly the multiset of cards each player
//! owns plus the supply (see `dominion_core::determinize`), so a
//! player-relative encoding loses nothing that matters: what any card effect
//! can react to is either "my cards", "the supply", or "the opponents' cards",
//! never a specific seat number. Encoding it that way also makes the feature
//! vector a fixed size regardless of how many opponents there are, which
//! keeps a 2-player and a 4-player position comparable to the network.

use dominion_core::card::NUM_CARDS;
use dominion_core::{Card, Ctx, Decision, GameState};

/// Counts are divided by this before being fed to the network, so a typical
/// value lands near 1.0 rather than in the tens.
const COUNT_SCALE: f32 = 10.0;

pub const FEATURE_DIM: usize = 4 * NUM_CARDS + 7;

/// Encode the position from `player`'s point of view.
///
/// Layout: own hand, own total holdings, remaining supply, opponents' total
/// holdings (summed across every other seat), then a handful of scalars.
/// All four card blocks are in the engine's canonical card order
/// (`Card::from_idx`), so index `i` always means the same card across calls.
pub fn encode(state: &GameState, player: usize, d: &Decision) -> [f32; FEATURE_DIM] {
    let mut f = [0.0f32; FEATURE_DIM];
    let me = &state.players[player];

    for &c in &me.hand {
        f[c.idx()] += 1.0;
    }
    for c in me.all_cards() {
        f[NUM_CARDS + c.idx()] += 1.0;
    }
    for i in 0..NUM_CARDS {
        f[2 * NUM_CARDS + i] = state.supply[i] as f32;
    }
    for (i, p) in state.players.iter().enumerate() {
        if i == player {
            continue;
        }
        for c in p.all_cards() {
            f[3 * NUM_CARDS + c.idx()] += 1.0;
        }
    }
    for x in f.iter_mut().take(4 * NUM_CARDS) {
        *x /= COUNT_SCALE;
    }

    let base = 4 * NUM_CARDS;
    f[base] = me.turns as f32 / 30.0;
    f[base + 1] = state.supply_of(Card::Province) as f32 / 8.0;
    f[base + 2] = me.coins as f32 / 10.0;
    f[base + 3] = me.actions as f32 / 5.0;
    f[base + 4] = me.buys as f32 / 5.0;
    f[base + 5] = state.empty_piles() as f32 / 3.0;
    f[base + 6] = if d.ctx == Ctx::BuyPhase { 1.0 } else { 0.0 };
    f
}
