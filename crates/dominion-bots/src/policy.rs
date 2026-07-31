//! Shared heuristic policy: a defensible default answer to every decision that
//! isn't "what do I buy".
//!
//! Nothing here is optimal — it is meant to be a solid, strategy-neutral
//! baseline that every bot inherits, so that two strategies differ only in
//! their buy menus, and so that search rollouts land in plausible positions
//! instead of nonsense ones.

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

/// Ranking used when *gaining* a card (Workshop, Remodel, Artisan, Mine).
/// Deliberately money-first: it is the fallback when a strategy has no opinion.
pub fn gain_preference(card: Card, state: &GameState, _player: usize) -> i32 {
    use Card::*;
    let pl = provinces_left(state);
    match card {
        Province => 1000,
        Gold => 900,
        // Duchies only once the game is nearly over.
        Duchy if pl <= 4 => 850,
        Silver => 700,
        Estate if pl <= 2 => 600,
        Curse => -1000,
        Estate | Duchy | Gardens => -100,
        Copper => -50,
        c => 400 + c.cost() as i32,
    }
}

/// Pick the best option for any decision, given a way to rank gains.
pub fn default_move_with(
    state: &GameState,
    d: &Decision,
    gain_rank: &dyn Fn(Card, &GameState, usize) -> i32,
) -> Move {
    let p = d.player;
    let player = &state.players[p];

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
            .max_by_key(|&c| gain_rank(c, state, p))
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
                Some(c) if gain_rank(c, state, p) > 0 => Move::Buy(c),
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
                .max_by_key(|&c| gain_rank(c, state, p));
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
            .max_by_key(|&c| gain_rank(c, state, p))
            .map(Move::Select)
            .unwrap_or(Move::Done),

        // Topdeck the card we most want to draw next turn.
        Ctx::ArtisanTopdeck => selects()
            .into_iter()
            .max_by_key(|&c| gain_rank(c, state, p))
            .map(Move::Select)
            .unwrap_or(Move::Done),
    }
}

/// How much a card gains from being played twice.
fn throne_value(card: Card) -> i32 {
    use Card::*;
    match card {
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
