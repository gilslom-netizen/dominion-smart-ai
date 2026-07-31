//! Turning the true game state into one consistent with what a player can
//! actually see.
//!
//! Dominion is a game of almost-perfect information: every card either player
//! owns is public knowledge, because every gain, trash and discard happens face
//! up. What is hidden is only *order* — which of my own cards are on top of my
//! deck, and which of the opponent's unseen cards are in their hand rather than
//! their deck.
//!
//! That makes determinization unusually clean and unusually effective here
//! compared to games like poker or bridge: we do not have to guess *what* cards
//! exist, only where they sit. Sampling a few orderings and searching each one
//! (perfect-information Monte Carlo) is therefore a much better approximation in
//! Dominion than it is in most hidden-information games.

use crate::card::Card;
use crate::rng::Rng;
use crate::state::GameState;

/// Produce a state consistent with everything `observer` legitimately knows,
/// with all remaining hidden information resampled.
///
/// Specifically:
/// * The observer's own deck order is reshuffled, except for the top
///   `known_top` cards they deliberately put there.
/// * Each opponent's hand and deck are pooled and redealt, since the observer
///   knows the multiset but not the split or the order.
/// * Everything else — supply, trash, play areas, the observer's hand and
///   discard — is public or already known, and is left untouched.
pub fn determinize(state: &GameState, observer: usize, rng: &mut Rng) -> GameState {
    let mut s = state.clone();

    for p in 0..s.players.len() {
        let pl = &mut s.players[p];
        if p == observer {
            // Own deck: the known top stays put, the rest is unknown order.
            let keep = (pl.known_top as usize).min(pl.deck.len());
            let split = pl.deck.len() - keep;
            rng.shuffle(&mut pl.deck[..split]);
        } else {
            // An opponent's hand and deck are one undifferentiated pool of
            // cards whose location we do not know.
            let hand_size = pl.hand.len();
            let mut pool: Vec<Card> = pl.deck.drain(..).collect();
            pool.append(&mut pl.hand);
            rng.shuffle(&mut pool);
            pl.hand = pool.split_off(pool.len() - hand_size.min(pool.len()));
            pl.deck = pool;
            // Whatever they knew about their own deck order, we do not.
            pl.known_top = 0;
        }
    }

    // Re-seed so that independent determinizations of the same position diverge
    // instead of replaying identical shuffles.
    s.rng = Rng::new(rng.next_u64());
    s
}

/// Check that a state could be the true state behind `observer`'s view of
/// `reference`: same public zones, same per-player card multisets.
///
/// Used by tests to prove determinization never invents or loses a card.
pub fn is_consistent_with(candidate: &GameState, reference: &GameState, observer: usize) -> bool {
    if candidate.supply != reference.supply || candidate.current != reference.current {
        return false;
    }
    if sorted(&candidate.trash) != sorted(&reference.trash) {
        return false;
    }
    for (a, b) in candidate.players.iter().zip(&reference.players) {
        // Public zones must match exactly.
        if sorted(&a.play) != sorted(&b.play) || sorted(&a.discard) != sorted(&b.discard) {
            return false;
        }
        // Hidden zones may be rearranged, but must hold the same cards.
        let mut hidden_a: Vec<Card> = a.deck.iter().chain(&a.hand).copied().collect();
        let mut hidden_b: Vec<Card> = b.deck.iter().chain(&b.hand).copied().collect();
        hidden_a.sort_unstable();
        hidden_b.sort_unstable();
        if hidden_a != hidden_b {
            return false;
        }
    }
    // The observer's own hand is known to them, so it must be untouched.
    sorted(&candidate.players[observer].hand) == sorted(&reference.players[observer].hand)
}

fn sorted(cards: &[Card]) -> Vec<Card> {
    let mut v = cards.to_vec();
    v.sort_unstable();
    v
}
