//! Declarative buy menus.
//!
//! A menu is an ordered list of rules; the first rule that matches an
//! affordable card wins. This is the format Geronimoo's simulator popularised
//! and it is expressive enough for every classic Base-set strategy, which makes
//! it the right shape for a benchmark ladder.

use dominion_core::{Card, Decision, GameState, Move};

use crate::policy;
use crate::Agent;

/// One line of a buy menu.
#[derive(Clone, Copy, Debug)]
pub struct BuyRule {
    pub card: Card,
    /// Stop buying once the player owns this many copies.
    pub max_owned: u32,
    /// Only while at least this many Provinces remain (green late, not early).
    pub min_provinces_left: u8,
    /// Only once the Province pile is down to this many (the endgame switch).
    pub max_provinces_left: u8,
}

impl BuyRule {
    pub fn new(card: Card) -> Self {
        BuyRule {
            card,
            max_owned: u32::MAX,
            min_provinces_left: 0,
            max_provinces_left: u8::MAX,
        }
    }
    pub fn at_most(mut self, n: u32) -> Self {
        self.max_owned = n;
        self
    }
    /// Only fire while more than `n` Provinces are left.
    pub fn while_provinces_above(mut self, n: u8) -> Self {
        self.min_provinces_left = n + 1;
        self
    }
    /// Only fire once `n` or fewer Provinces are left.
    pub fn when_provinces_at_most(mut self, n: u8) -> Self {
        self.max_provinces_left = n;
        self
    }

    fn matches(&self, card: Card, state: &GameState, player: usize) -> bool {
        if card != self.card {
            return false;
        }
        let left = state.supply_of(Card::Province);
        if left < self.min_provinces_left || left > self.max_provinces_left {
            return false;
        }
        (state.players[player].count_owned(card) as u32) < self.max_owned
    }
}

#[derive(Clone, Debug)]
pub struct BuyMenu {
    pub name: String,
    pub rules: Vec<BuyRule>,
}

impl BuyMenu {
    pub fn new(name: &str, rules: Vec<BuyRule>) -> Self {
        BuyMenu {
            name: name.into(),
            rules,
        }
    }

    /// Score a card as a gain target. Cards the menu does not want score
    /// negative, so the buy phase declines them — but a forced gain (Workshop,
    /// Remodel) still picks the least bad option.
    pub fn rank(&self, card: Card, state: &GameState, player: usize) -> i32 {
        match self
            .rules
            .iter()
            .position(|r| r.matches(card, state, player))
        {
            Some(i) => 100_000 - (i as i32) * 100,
            None => policy::gain_preference(card, state, player) - 100_000,
        }
    }
}

/// An agent that buys by menu and defers every other decision to the shared
/// heuristic policy.
pub struct MenuBot {
    pub menu: BuyMenu,
}

impl MenuBot {
    pub fn new(menu: BuyMenu) -> Self {
        MenuBot { menu }
    }
}

impl Agent for MenuBot {
    fn decide(&mut self, state: &GameState, d: &Decision) -> Move {
        let menu = &self.menu;
        policy::default_move_with(state, d, &|c, s, p| menu.rank(c, s, p))
    }
    fn name(&self) -> String {
        self.menu.name.clone()
    }
}

// ---------------------------------------------------------------- strategies

fn r(card: Card) -> BuyRule {
    BuyRule::new(card)
}

/// The canonical benchmark: Province / Gold / Silver with a late green switch.
/// Anything that cannot beat this is not a Dominion AI.
pub fn big_money() -> BuyMenu {
    use Card::*;
    BuyMenu::new(
        "BigMoney",
        vec![
            r(Province),
            r(Gold),
            r(Duchy).when_provinces_at_most(4),
            r(Estate).when_provinces_at_most(2),
            r(Silver),
        ],
    )
}

/// Big Money plus a single Smithy — the standard "can your bot play a card"
/// bar, and a clear step above plain Big Money.
pub fn big_money_smithy() -> BuyMenu {
    use Card::*;
    BuyMenu::new(
        "BM+Smithy",
        vec![
            r(Province),
            r(Gold),
            r(Duchy).when_provinces_at_most(4),
            r(Estate).when_provinces_at_most(2),
            r(Smithy).at_most(1),
            r(Silver),
        ],
    )
}

/// Two Witches on top of money. Cursing is the strongest single effect in the
/// Base set, so this is a real test of whether an agent handles junk.
pub fn double_witch() -> BuyMenu {
    use Card::*;
    BuyMenu::new(
        "DoubleWitch",
        vec![
            r(Province),
            r(Gold),
            r(Witch).at_most(2),
            r(Duchy).when_provinces_at_most(4),
            r(Estate).when_provinces_at_most(2),
            r(Silver),
        ],
    )
}

/// Money with a Militia to slow the opponent down.
pub fn militia_money() -> BuyMenu {
    use Card::*;
    BuyMenu::new(
        "MilitiaMoney",
        vec![
            r(Province),
            r(Gold),
            r(Duchy).when_provinces_at_most(4),
            r(Estate).when_provinces_at_most(2),
            r(Militia).at_most(1),
            r(Silver),
        ],
    )
}

/// Chapel thins the deck early, then plays money. Punishes an engine that
/// cannot handle a fast, small deck.
pub fn chapel_money() -> BuyMenu {
    use Card::*;
    BuyMenu::new(
        "ChapelMoney",
        vec![
            r(Province),
            r(Gold),
            r(Chapel).at_most(1).while_provinces_above(5),
            r(Duchy).when_provinces_at_most(4),
            r(Estate).when_provinces_at_most(2),
            r(Silver),
        ],
    )
}

/// A cheap draw engine: Villages and Smithies with Markets for economy.
pub fn village_smithy() -> BuyMenu {
    use Card::*;
    BuyMenu::new(
        "VillageSmithy",
        vec![
            r(Province),
            r(Gold),
            r(Duchy).when_provinces_at_most(3),
            r(Estate).when_provinces_at_most(2),
            r(Smithy).at_most(3),
            r(Village).at_most(5),
            r(Silver),
        ],
    )
}

/// Every named strategy, for round-robin benchmarking.
pub fn ladder() -> Vec<BuyMenu> {
    vec![
        big_money(),
        big_money_smithy(),
        double_witch(),
        militia_money(),
        chapel_money(),
        village_smithy(),
    ]
}

/// The kingdom cards a menu needs in the supply to make sense.
pub fn required_kingdom(menu: &BuyMenu) -> Vec<Card> {
    menu.rules
        .iter()
        .map(|r| r.card)
        .filter(|c| dominion_core::KINGDOM_CARDS.contains(c))
        .collect()
}
