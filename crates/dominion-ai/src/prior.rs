//! Move priors and domain-safe move restriction.
//!
//! Plain UCT is close to useless in Dominion. The branching factor at a buy
//! decision is ~15, adjacent buys differ by around a percentage point of win
//! probability, and a rollout returns a single bit. With any affordable budget
//! the visit counts come out flat and the search picks noise.
//!
//! The fix is the same one that makes AlphaZero-style search work: bias
//! exploration with a prior over moves, so the budget is spent on plausible
//! lines. This module is the prior. It is heuristic today and a policy network
//! later — the search does not care which, as long as it gets a probability
//! distribution over the legal moves.

use dominion_bots::policy;
use dominion_core::{Card, Ctx, Decision, GameState, Move};

/// Cut moves that are provably or near-provably wrong, before search sees them.
///
/// The one hard rule here is Base-set specific and worth stating: **playing a
/// Treasure during the buy phase is never a mistake**. The Base set has no
/// Treasure with a downside and no card that cares how much money is unspent,
/// and the order treasures are played in cannot matter (Merchant pays out on
/// the first Silver either way). So whenever a Treasure can be played, playing
/// one is at least as good as anything else, and everything else can go.
///
/// That single restriction removes most of the buy-phase branching, which is
/// where the search budget was being wasted.
pub fn restrict(state: &GameState, d: &Decision) -> Vec<Move> {
    if d.ctx == Ctx::BuyPhase {
        let treasures: Vec<Move> = d
            .options
            .iter()
            .copied()
            .filter(|m| matches!(m, Move::Play(c) if c.is_treasure()))
            .collect();
        // The order treasures are played in cannot matter either, so collapse
        // the choice to one move and let the engine auto-resolve it. This turns
        // four searched decisions per turn into zero.
        if let Some(&first) = treasures.first() {
            return vec![first];
        }
    }

    let filtered: Vec<Move> = d
        .options
        .iter()
        .copied()
        .filter(|m| match m {
            // Deliberately taking a Curse is never right in the Base set.
            Move::Buy(Card::Curse) => false,
            // Buying Copper only makes sense for Gardens piles, which the
            // prior can still reach through the Select path if it matters.
            Move::Buy(Card::Copper) => state.in_supply[Card::Gardens.idx()],
            _ => true,
        })
        .collect();

    if filtered.is_empty() {
        d.options.clone()
    } else {
        filtered
    }
}

/// A probability distribution over `options`, summing to 1.
///
/// The shape is deliberately simple: the heuristic's own choice gets most of
/// the mass, obviously poor moves get almost none, and everything else shares
/// the rest. That is enough to make the search look at good moves first while
/// leaving it free to disagree.
pub fn priors(state: &GameState, d: &Decision, options: &[Move]) -> Vec<f32> {
    let heuristic = policy::default_move(state, d);
    let player = &state.players[d.player];
    let provinces = state.supply_of(Card::Province);

    let mut w: Vec<f32> = options
        .iter()
        .map(|&m| {
            let mut s: f32 = 1.0;
            match m {
                Move::Buy(Card::Curse) => s = 0.02,
                Move::Buy(Card::Copper) => s = 0.05,
                // Greening early loses the game more often than any single buy.
                Move::Buy(c) if c.is_victory() && c != Card::Province && provinces > 4 => {
                    s = 0.1;
                }
                // Passing on a buy with real money in hand is almost never right.
                Move::Done if d.ctx == Ctx::BuyPhase && player.buys > 0 && player.coins >= 3 => {
                    s = 0.1;
                }
                _ => {}
            }
            if m == heuristic {
                s += 6.0;
            }
            s
        })
        .collect();

    let total: f32 = w.iter().sum();
    if total <= 0.0 {
        let u = 1.0 / options.len() as f32;
        return vec![u; options.len()];
    }
    for x in &mut w {
        *x /= total;
    }
    w
}
