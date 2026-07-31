//! Card definitions for Dominion Base set, 2nd edition.
//!
//! 7 basic cards + 26 kingdom cards = 33 distinct cards. The discriminants are
//! stable and used directly as indices into supply/count arrays, so do not
//! reorder them.

use std::fmt;

pub const NUM_CARDS: usize = 33;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Card {
    // Basic supply
    Copper = 0,
    Silver,
    Gold,
    Estate,
    Duchy,
    Province,
    Curse,
    // Kingdom ($2)
    Cellar,
    Chapel,
    Moat,
    // ($3)
    Harbinger,
    Merchant,
    Vassal,
    Village,
    Workshop,
    // ($4)
    Bureaucrat,
    Gardens,
    Militia,
    Moneylender,
    Poacher,
    Remodel,
    Smithy,
    ThroneRoom,
    // ($5)
    Bandit,
    CouncilRoom,
    Festival,
    Laboratory,
    Library,
    Market,
    Mine,
    Sentry,
    Witch,
    // ($6)
    Artisan,
}

pub const ALL_CARDS: [Card; NUM_CARDS] = {
    use Card::*;
    [
        Copper,
        Silver,
        Gold,
        Estate,
        Duchy,
        Province,
        Curse,
        Cellar,
        Chapel,
        Moat,
        Harbinger,
        Merchant,
        Vassal,
        Village,
        Workshop,
        Bureaucrat,
        Gardens,
        Militia,
        Moneylender,
        Poacher,
        Remodel,
        Smithy,
        ThroneRoom,
        Bandit,
        CouncilRoom,
        Festival,
        Laboratory,
        Library,
        Market,
        Mine,
        Sentry,
        Witch,
        Artisan,
    ]
};

/// The 26 kingdom cards of Base 2E, in supply order.
pub const KINGDOM_CARDS: [Card; 26] = {
    use Card::*;
    [
        Cellar,
        Chapel,
        Moat,
        Harbinger,
        Merchant,
        Vassal,
        Village,
        Workshop,
        Bureaucrat,
        Gardens,
        Militia,
        Moneylender,
        Poacher,
        Remodel,
        Smithy,
        ThroneRoom,
        Bandit,
        CouncilRoom,
        Festival,
        Laboratory,
        Library,
        Market,
        Mine,
        Sentry,
        Witch,
        Artisan,
    ]
};

pub const BASIC_CARDS: [Card; 7] = {
    use Card::*;
    [Copper, Silver, Gold, Estate, Duchy, Province, Curse]
};

/// Card type line. A card can carry several types (e.g. Moat is
/// Action-Reaction).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Types(u8);

impl Types {
    pub const ACTION: Types = Types(1 << 0);
    pub const TREASURE: Types = Types(1 << 1);
    pub const VICTORY: Types = Types(1 << 2);
    pub const CURSE: Types = Types(1 << 3);
    pub const ATTACK: Types = Types(1 << 4);
    pub const REACTION: Types = Types(1 << 5);

    #[inline]
    pub const fn bits(self) -> u8 {
        self.0
    }
    #[inline]
    pub const fn from_bits_truncate(bits: u8) -> Self {
        Types(bits)
    }
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

impl std::ops::BitOr for Types {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Types(self.0 | rhs.0)
    }
}

impl Card {
    #[inline]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn from_idx(i: usize) -> Card {
        ALL_CARDS[i]
    }

    pub const fn cost(self) -> u8 {
        use Card::*;
        match self {
            Copper | Curse => 0,
            Estate | Cellar | Chapel | Moat => 2,
            Silver | Harbinger | Merchant | Vassal | Village | Workshop => 3,
            Duchy | Bureaucrat | Gardens | Militia | Moneylender | Poacher | Remodel | Smithy
            | ThroneRoom => 4,
            Gold | Bandit | CouncilRoom | Festival | Laboratory | Library | Market | Mine
            | Sentry | Witch => 5,
            Artisan => 6,
            Province => 8,
        }
    }

    pub const fn types(self) -> Types {
        use Card::*;
        let bits = match self {
            Copper | Silver | Gold => Types::TREASURE.bits(),
            Estate | Duchy | Province | Gardens => Types::VICTORY.bits(),
            Curse => Types::CURSE.bits(),
            Moat => Types::ACTION.bits() | Types::REACTION.bits(),
            Bureaucrat | Militia | Bandit | Witch => Types::ACTION.bits() | Types::ATTACK.bits(),
            _ => Types::ACTION.bits(),
        };
        Types::from_bits_truncate(bits)
    }

    #[inline]
    pub const fn is_action(self) -> bool {
        self.types().contains(Types::ACTION)
    }
    #[inline]
    pub const fn is_treasure(self) -> bool {
        self.types().contains(Types::TREASURE)
    }
    #[inline]
    pub const fn is_victory(self) -> bool {
        self.types().contains(Types::VICTORY)
    }
    #[inline]
    pub const fn is_curse(self) -> bool {
        self.types().contains(Types::CURSE)
    }
    #[inline]
    pub const fn is_attack(self) -> bool {
        self.types().contains(Types::ATTACK)
    }
    #[inline]
    pub const fn is_reaction(self) -> bool {
        self.types().contains(Types::REACTION)
    }

    /// Coins produced when played as a Treasure.
    pub const fn coin_value(self) -> u8 {
        use Card::*;
        match self {
            Copper => 1,
            Silver => 2,
            Gold => 3,
            _ => 0,
        }
    }

    /// Static victory points. Gardens is variable and handled by the scorer.
    pub const fn static_vp(self) -> i32 {
        use Card::*;
        match self {
            Estate => 1,
            Duchy => 3,
            Province => 6,
            Curse => -1,
            _ => 0,
        }
    }

    pub const fn name(self) -> &'static str {
        use Card::*;
        match self {
            Copper => "Copper",
            Silver => "Silver",
            Gold => "Gold",
            Estate => "Estate",
            Duchy => "Duchy",
            Province => "Province",
            Curse => "Curse",
            Cellar => "Cellar",
            Chapel => "Chapel",
            Moat => "Moat",
            Harbinger => "Harbinger",
            Merchant => "Merchant",
            Vassal => "Vassal",
            Village => "Village",
            Workshop => "Workshop",
            Bureaucrat => "Bureaucrat",
            Gardens => "Gardens",
            Militia => "Militia",
            Moneylender => "Moneylender",
            Poacher => "Poacher",
            Remodel => "Remodel",
            Smithy => "Smithy",
            ThroneRoom => "Throne Room",
            Bandit => "Bandit",
            CouncilRoom => "Council Room",
            Festival => "Festival",
            Laboratory => "Laboratory",
            Library => "Library",
            Market => "Market",
            Mine => "Mine",
            Sentry => "Sentry",
            Witch => "Witch",
            Artisan => "Artisan",
        }
    }

    /// Parse a card by name. Accepts the canonical name with or without
    /// spaces, case-insensitively ("Throne Room", "throneroom", "THRONE ROOM").
    pub fn parse(s: &str) -> Option<Card> {
        let norm = |t: &str| -> String {
            t.chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .map(|c| c.to_ascii_lowercase())
                .collect()
        };
        let want = norm(s);
        ALL_CARDS.into_iter().find(|c| norm(c.name()) == want)
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A multiset of cards, stored as counts indexed by [`Card::idx`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CardCounts(pub [u8; NUM_CARDS]);

impl Default for CardCounts {
    fn default() -> Self {
        Self::new()
    }
}

impl CardCounts {
    pub fn new() -> Self {
        Self([0; NUM_CARDS])
    }

    pub fn from_slice(cards: &[Card]) -> Self {
        let mut c = Self::new();
        for &card in cards {
            c.0[card.idx()] += 1;
        }
        c
    }

    #[inline]
    pub fn get(&self, card: Card) -> u8 {
        self.0[card.idx()]
    }

    #[inline]
    pub fn add(&mut self, card: Card, n: u8) {
        self.0[card.idx()] += n;
    }

    #[inline]
    pub fn total(&self) -> u32 {
        self.0.iter().map(|&n| n as u32).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Card, u8)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, &n)| n > 0)
            .map(|(i, &n)| (Card::from_idx(i), n))
    }
}
