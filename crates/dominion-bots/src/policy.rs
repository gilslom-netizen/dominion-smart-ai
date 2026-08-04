//! Shared heuristic policy: a defensible default answer to every decision that
//! isn't "what do I buy".
//!
//! Nothing here is optimal — it is meant to be a solid, strategy-neutral
//! baseline that every bot inherits, so that two strategies differ only in
//! their buy menus, and so that search rollouts land in plausible positions
//! instead of nonsense ones.

use dominion_core::card::NUM_CARDS;
use dominion_core::{Card, Ctx, Decision, GameState, Move, PlayerState, Rng};

use crate::Agent;

/// Higher plays first. The ordering follows the usual rule of thumb: cards
/// that replace the Action they cost come before cards that don't.
pub fn action_priority(card: Card, hand: &[Card], _state: &GameState) -> i32 {
    use Card::*;
    let other_actions = hand.iter().filter(|c| c.is_action() && **c != card).count();
    match card {
        // +2 Actions: always safe, always first.
        Village => 100,
        Festival => 95,
        // Cantrips: net-neutral on Actions, so they can only help.
        Laboratory => 92,
        Market => 90,
        Merchant => 88,
        Poacher => 87,
        // Sentry sifts the deck, so it wants to go before any draw.
        Sentry => 94,
        Harbinger => 86,
        // Cellar is a cantrip, but only worth it with junk to dump.
        Cellar => {
            if hand.iter().any(|c| c.is_victory() || c.is_curse()) {
                85
            } else {
                5
            }
        }
        // Throne Room is terminal, but doubling something beats playing it
        // straight, so it leads the terminals — unless there is nothing to
        // double, in which case it is a dead card.
        ThroneRoom => {
            if other_actions > 0 {
                80
            } else {
                1
            }
        }
        // Terminals, roughly by impact.
        Chapel => 75,
        Witch => 70,
        CouncilRoom => 65,
        Library => 62,
        Smithy => 60,
        Militia => 58,
        Moneylender => {
            if hand.contains(&Copper) {
                57
            } else {
                2
            }
        }
        Vassal => 56,
        Bandit => 55,
        Workshop => 52,
        Mine => 50,
        Remodel => 48,
        Bureaucrat => 45,
        _ => 0,
    }
}

/// Lower is discarded first. Junk goes, economy stays.
pub fn discard_rank(card: Card) -> i32 {
    use Card::*;
    match card {
        Curse => 0,
        Estate | Gardens => 1,
        Duchy => 2,
        Province => 3,
        Copper => 4,
        c if c.is_action() => 5,
        Silver => 8,
        Gold => 9,
        _ => 6,
    }
}

/// Lower is trashed first. Cards not worth trashing score high.
pub fn trash_rank(card: Card) -> i32 {
    use Card::*;
    match card {
        Curse => 0,
        Estate => 1,
        Copper => 2,
        _ => 100,
    }
}

/// Total coins the player's deck can produce, used to stop Chapel from
/// trashing the economy away entirely.
fn deck_coin_value(p: &PlayerState) -> u32 {
    p.all_cards().map(|c| c.coin_value() as u32).sum()
}

/// How many Provinces are left, the standard endgame clock.
fn provinces_left(state: &GameState) -> u8 {
    state.supply_of(Card::Province)
}

/// Is this card worth trashing right now?
fn worth_trashing(card: Card, state: &GameState, p: usize) -> bool {
    let player = &state.players[p];
    match card {
        Card::Curse => true,
        // Estates are victory points once the endgame is in sight.
        Card::Estate => provinces_left(state) > 4,
        // Keep enough economy to still hit $5-$6.
        Card::Copper => deck_coin_value(player) > 3,
        _ => false,
    }
}

/// A summary of what a player's deck is made of, computed once per decision
/// rather than once per candidate card.
#[derive(Clone, Copy, Debug)]
pub struct DeckStats {
    pub total: u32,
    /// How many copies of each card the player owns, indexed by `Card::idx`.
    /// Precomputed because the ranking asks about half a dozen cards per
    /// candidate, and walking the deck for each one made the rollout — and so
    /// the whole search — more than twice as slow.
    pub owned: [u8; NUM_CARDS],
    /// Actions that consume the turn's only Action and give none back.
    pub terminals: u32,
    /// Actions granting +2 Actions, which is what makes terminals stackable.
    pub villages: u32,
    pub coin_value: u32,
    /// Whether any opponent owns an Attack, the only reason to want a Moat.
    pub under_attack: bool,
}

impl Default for DeckStats {
    fn default() -> Self {
        DeckStats {
            total: 0,
            owned: [0; NUM_CARDS],
            terminals: 0,
            villages: 0,
            coin_value: 0,
            under_attack: false,
        }
    }
}

impl DeckStats {
    pub fn of(state: &GameState, player: usize) -> Self {
        use Card::*;
        let mut st = DeckStats::default();
        for c in state.players[player].all_cards() {
            st.total += 1;
            st.owned[c.idx()] = st.owned[c.idx()].saturating_add(1);
            st.coin_value += c.coin_value() as u32;
            match c {
                Village | Festival => st.villages += 1,
                // Cantrips replace the Action they cost, so they are free.
                Market | Laboratory | Merchant | Poacher | Harbinger | Cellar | Sentry => {}
                c if c.is_action() => st.terminals += 1,
                _ => {}
            }
        }
        st.under_attack = state
            .players
            .iter()
            .enumerate()
            .any(|(i, p)| i != player && p.all_cards().any(|c| c.is_attack()));
        st
    }

    #[inline]
    pub fn count(&self, card: Card) -> i32 {
        self.owned[card.idx()] as i32
    }

    #[inline]
    pub fn coppers(&self) -> u32 {
        self.owned[Card::Copper.idx()] as u32
    }

    /// How many more terminals the deck can absorb before they start colliding.
    /// One Action per turn, plus two per Village.
    pub fn terminal_room(&self) -> i32 {
        1 + 2 * self.villages as i32 - self.terminals as i32
    }
}

/// Ranking used for every gain: buying, but also Workshop, Remodel, Artisan and
/// Mine.
///
/// This is the single most important heuristic in the project. It is the buy
/// policy of the baseline bots, the prior that steers the search, *and* the
/// rollout policy that assigns value to leaf positions — so a version that only
/// ever buys money makes the search structurally blind to engines, no matter
/// how many iterations it runs.
pub fn gain_preference(card: Card, state: &GameState, _player: usize, st: &DeckStats) -> i32 {
    use Card::*;
    let pl = provinces_left(state);
    let owned = |c: Card| st.count(c);
    let room = st.terminal_room();
    let _ = pl > 5;

    // A deck has room for very few terminals, and they are not interchangeable.
    // Buying a $4 Smithy on turn two uses up the only slot and locks out the
    // Witch the deck really wanted — which is exactly how an earlier version
    // managed to lose 72% to Double Witch while ranking Witch above Gold.
    // So the slot is reserved until the Witches are bought.
    let saving_for_witch = state.in_supply[Witch.idx()]
        && state.supply_of(Witch) > 0
        && owned(Witch) < 2
        // Two Witches is standard even with no Village to stack them on: the
        // curse output is worth the occasional collision.
        && room >= 0;

    // The ranking is calibrated against the benchmark ladder, not against
    // intuition. An earlier version that bought a wide spread of engine pieces
    // scored *worse* than plain Big Money + Smithy, because a deck with one of
    // everything draws none of them. What survives measurement is: money as the
    // spine, cursing above Gold, and exactly as many terminals as the deck can
    // actually play.
    match card {
        Province => 1000,
        Curse => -1000,

        // --- endgame greening --------------------------------------------
        Duchy if pl <= 4 => 880,
        Estate if pl <= 2 => 700,

        // --- the few cards worth more than money -------------------------
        // Handing out Curses is the strongest effect in the Base set.
        Witch if owned(Witch) < 2 && room >= 0 => 960,
        Gold => 900,
        // A cantrip that draws two: strictly better than the Silver it replaces.
        Laboratory if room >= 0 => 890,
        // One terminal draw is the classic Big Money upgrade; a second one
        // without Villages just collides with the first.
        Smithy if room > 0 && !saving_for_witch && owned(Smithy) < 1 => 860,
        Militia if room > 0 && !saving_for_witch && owned(Militia) < 1 => 835,
        CouncilRoom if room > 0 && !saving_for_witch && owned(CouncilRoom) < 1 && owned(Smithy) == 0 => 830,
        // Cantrips add economy without diluting the draw.
        Market if owned(Market) < 3 => 760,

        Silver => 700,

        // Villages only once terminals are genuinely colliding.
        Village if room < 0 => 690,
        Festival if room < 0 => 680,
        // Only bother with a Moat if somebody is actually attacking.
        Moat if room > 0 && owned(Moat) < 1 && st.under_attack => 670,

        // --- things we do not want ----------------------------------------
        Estate | Duchy | Gardens => -100,
        Copper => -50,
        // Everything else is playable but worse than a Silver.
        c if c.is_action() => 300 + c.cost() as i32,
        _ => 100,
    }
}

/// Pick the best option for any decision, given a way to rank gains.
pub fn default_move_with(
    state: &GameState,
    d: &Decision,
    gain_rank: &dyn Fn(Card, &GameState, usize, &DeckStats) -> i32,
) -> Move {
    let p = d.player;
    let player = &state.players[p];
    let stats = DeckStats::of(state, d.player);

    // Convenience closures over the offered options.
    let selects = || -> Vec<Card> {
        d.options
            .iter()
            .filter_map(|m| match m {
                Move::Select(c) => Some(*c),
                _ => None,
            })
            .collect()
    };
    let best_gain = |cards: Vec<Card>| -> Option<Card> {
        cards
            .into_iter()
            .max_by_key(|&c| gain_rank(c, state, p, &stats))
    };

    match d.ctx {
        Ctx::ActionPhase => d
            .options
            .iter()
            .filter_map(|m| match m {
                Move::Play(c) => Some((*c, action_priority(*c, &player.hand, state))),
                _ => None,
            })
            .filter(|(_, prio)| *prio > 3)
            .max_by_key(|(_, prio)| *prio)
            .map(|(c, _)| Move::Play(c))
            .unwrap_or(Move::Done),

        Ctx::BuyPhase => {
            // Always cash in first; the buy menu decides the rest.
            if let Some(m) = d.options.iter().find(|m| matches!(m, Move::Play(_))) {
                return *m;
            }
            let buys: Vec<Card> = d
                .options
                .iter()
                .filter_map(|m| match m {
                    Move::Buy(c) => Some(*c),
                    _ => None,
                })
                .collect();
            match best_gain(buys) {
                Some(c) if gain_rank(c, state, p, &stats) > 0 => Move::Buy(c),
                _ => Move::Done,
            }
        }

        // Blocking an attack is free and never wrong in the Base set.
        Ctx::MoatReveal => Move::Select(Card::Moat),

        Ctx::CellarDiscard => {
            let junk = selects()
                .into_iter()
                .filter(|c| c.is_victory() || c.is_curse())
                .min_by_key(|&c| discard_rank(c));
            junk.map(Move::Select).unwrap_or(Move::Done)
        }

        Ctx::ChapelTrash => selects()
            .into_iter()
            .filter(|&c| worth_trashing(c, state, p))
            .min_by_key(|&c| trash_rank(c))
            .map(Move::Select)
            .unwrap_or(Move::Done),

        Ctx::MoneylenderTrash => {
            if worth_trashing(Card::Copper, state, p) {
                Move::Select(Card::Copper)
            } else {
                Move::Done
            }
        }

        Ctx::MilitiaDiscard | Ctx::PoacherDiscard => selects()
            .into_iter()
            .min_by_key(|&c| discard_rank(c))
            .map(Move::Select)
            .unwrap_or(Move::Done),

        // Put the most useful card back on top of the deck.
        Ctx::HarbingerTopdeck => {
            let best = selects()
                .into_iter()
                .filter(|c| !c.is_victory() && !c.is_curse())
                .max_by_key(|&c| gain_rank(c, state, p, &stats));
            best.map(Move::Select).unwrap_or(Move::Done)
        }

        // Free extra play: take it unless the card is dead weight.
        Ctx::VassalPlay => d
            .options
            .iter()
            .filter_map(|m| match m {
                Move::Play(c) => Some(*c),
                _ => None,
            })
            .find(|&c| action_priority(c, &player.hand, state) > 3)
            .map(Move::Play)
            .unwrap_or(Move::Done),

        Ctx::WorkshopGain | Ctx::RemodelGain | Ctx::MineGain | Ctx::ArtisanGain => {
            best_gain(selects()).map(Move::Select).unwrap_or(Move::Done)
        }

        Ctx::RemodelTrash => {
            // Trash junk if there is any; otherwise upgrade the cheapest card
            // that actually buys something better.
            let opts = selects();
            let junk = opts
                .iter()
                .copied()
                .filter(|&c| worth_trashing(c, state, p))
                .min_by_key(|&c| trash_rank(c));
            junk.or_else(|| opts.iter().copied().min_by_key(|c| c.cost()))
                .map(Move::Select)
                .unwrap_or(Move::Done)
        }

        Ctx::MineTrash => {
            let opts = selects();
            // Upgrading Silver into Gold is worth more than Copper into Silver.
            let pick = opts
                .iter()
                .copied()
                .find(|&c| c == Card::Silver && state.supply_of(Card::Gold) > 0)
                .or_else(|| {
                    opts.iter()
                        .copied()
                        .find(|&c| c == Card::Copper && state.supply_of(Card::Silver) > 0)
                });
            pick.map(Move::Select).unwrap_or(Move::Done)
        }

        Ctx::ThroneRoomPlay => d
            .options
            .iter()
            .filter_map(|m| match m {
                Move::Play(c) => Some((*c, action_priority(*c, &player.hand, state))),
                _ => None,
            })
            .filter(|(_, prio)| *prio > 3)
            // Doubling a terminal draw beats doubling a Village.
            .max_by_key(|(c, prio)| (throne_value(*c), *prio))
            .map(|(c, _)| Move::Play(c))
            .unwrap_or(Move::Done),

        // Keep Actions we can still use; set the rest aside.
        Ctx::LibrarySetAside => {
            let card = selects().first().copied();
            match card {
                Some(c) if player.actions > 0 && action_priority(c, &player.hand, state) > 3 => {
                    Move::Done
                }
                Some(c) => Move::Select(c),
                None => Move::Done,
            }
        }

        // Reveal the cheapest Victory card: keep the good ones in hand.
        Ctx::BureaucratReveal => selects()
            .into_iter()
            .min_by_key(|c| c.cost())
            .map(Move::Select)
            .unwrap_or(Move::Done),

        // As the victim, trash the cheaper of the two treasures.
        Ctx::BanditTrash => selects()
            .into_iter()
            .min_by_key(|c| c.cost())
            .map(Move::Select)
            .unwrap_or(Move::Done),

        Ctx::SentryTrash => selects()
            .into_iter()
            .filter(|&c| worth_trashing(c, state, p))
            .min_by_key(|&c| trash_rank(c))
            .map(Move::Select)
            .unwrap_or(Move::Done),

        Ctx::SentryDiscard => selects()
            .into_iter()
            .filter(|c| c.is_victory() || c.is_curse())
            .min_by_key(|&c| discard_rank(c))
            .map(Move::Select)
            .unwrap_or(Move::Done),

        // Draw the better card first.
        Ctx::SentryOrder => selects()
            .into_iter()
            .max_by_key(|&c| gain_rank(c, state, p, &stats))
            .map(Move::Select)
            .unwrap_or(Move::Done),

        // Topdeck the card we most want to draw next turn.
        Ctx::ArtisanTopdeck => selects()
            .into_iter()
            .max_by_key(|&c| gain_rank(c, state, p, &stats))
            .map(Move::Select)
            .unwrap_or(Move::Done),
    }
}

/// How much a card gains from being played twice.
fn throne_value(card: Card) -> i32 {
    use Card::*;
    match card {
        // Vassal doubled is +$4 and two cards off the top of the deck played
        // if they are Actions, which chains in an Action-dense deck. Ranking
        // it last made Throne Room + Vassal unplayable by this policy, so the
        // strategy could never be measured — only the refusal could.
        Vassal => 4,
        Witch | CouncilRoom | Smithy | Laboratory | Market | Bandit => 3,
        Militia | Festival | Village | Poacher | Merchant => 2,
        _ => 1,
    }
}

/// The shared heuristic with the default money-first gain ranking.
pub fn default_move(state: &GameState, d: &Decision) -> Move {
    default_move_with(state, d, &gain_preference)
}

/// Uniformly random legal moves. Useful as a sanity opponent and to confirm
/// that a stronger agent is actually doing something.
pub struct RandomAgent {
    rng: Rng,
}

impl RandomAgent {
    pub fn new(seed: u64) -> Self {
        RandomAgent { rng: Rng::new(seed) }
    }
}

impl Agent for RandomAgent {
    fn decide(&mut self, _state: &GameState, d: &Decision) -> Move {
        d.options[self.rng.below(d.options.len() as u64) as usize]
    }
    fn name(&self) -> String {
        "Random".into()
    }
}

/// The shared heuristic as a bot in its own right, with no buy menu on top.
///
/// It is the honest baseline for the search agents, since it is exactly the
/// policy their rollouts use: any strength the search shows over this bot comes
/// from the search itself, not from a better hand-written strategy.
pub struct HeuristicBot;

impl Agent for HeuristicBot {
    fn decide(&mut self, state: &GameState, d: &Decision) -> Move {
        default_move(state, d)
    }
    fn name(&self) -> String {
        "Heuristic".into()
    }
}
