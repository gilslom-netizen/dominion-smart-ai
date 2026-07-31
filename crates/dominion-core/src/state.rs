//! Game state, moves and the continuation stack.
//!
//! The engine is a state machine: every card effect is expressed as a sequence
//! of [`Frame`]s pushed onto [`GameState::stack`]. The engine runs frames until
//! one of them needs input, at which point it parks a [`Decision`] and returns.
//! That means the whole game state is a plain value that can be cloned or
//! serialized at *any* decision point — which is what search and log replay
//! both need.

use crate::card::{Card, NUM_CARDS};
use crate::rng::Rng;

pub const MAX_PLAYERS: usize = 4;

/// A single choice offered to a player.
///
/// Compound choices ("discard any number of cards") are decomposed into a loop
/// of single-card choices terminated by [`Move::Done`], which keeps the action
/// space small and uniform.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Move {
    /// Play a card from hand (or, for Vassal/Throne Room, the card in question).
    Play(Card),
    /// Buy a card from the supply.
    Buy(Card),
    /// Pick a card in the current selection context (discard it, trash it,
    /// gain it, topdeck it — the context says which).
    Select(Card),
    /// End the current phase, or stop selecting / decline an optional effect.
    Done,
}

/// Size of the flat move space used by policy networks.
pub const MOVE_SPACE: usize = 3 * NUM_CARDS + 1;

impl Move {
    /// Dense index in `[0, MOVE_SPACE)`, stable across games.
    #[inline]
    pub fn index(self) -> usize {
        match self {
            Move::Play(c) => c.idx(),
            Move::Buy(c) => NUM_CARDS + c.idx(),
            Move::Select(c) => 2 * NUM_CARDS + c.idx(),
            Move::Done => 3 * NUM_CARDS,
        }
    }

    pub fn from_index(i: usize) -> Option<Move> {
        match i {
            i if i < NUM_CARDS => Some(Move::Play(Card::from_idx(i))),
            i if i < 2 * NUM_CARDS => Some(Move::Buy(Card::from_idx(i - NUM_CARDS))),
            i if i < 3 * NUM_CARDS => Some(Move::Select(Card::from_idx(i - 2 * NUM_CARDS))),
            i if i == 3 * NUM_CARDS => Some(Move::Done),
            _ => None,
        }
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Move::Play(c) => write!(f, "play {c}"),
            Move::Buy(c) => write!(f, "buy {c}"),
            Move::Select(c) => write!(f, "pick {c}"),
            Move::Done => write!(f, "done"),
        }
    }
}

/// What the pending decision is about. Carries enough context for a human, a
/// heuristic bot or a network to interpret the offered [`Move`]s.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Ctx {
    /// Play an Action, or end the action phase.
    ActionPhase,
    /// Play a Treasure, buy a card, or end the turn.
    BuyPhase,
    /// Reveal Moat to block an incoming attack.
    MoatReveal,
    CellarDiscard,
    ChapelTrash,
    /// Put a card from the discard pile onto the deck.
    HarbingerTopdeck,
    /// Play the Action that Vassal just discarded.
    VassalPlay,
    WorkshopGain,
    /// Reveal a Victory card from hand to put on top of the deck.
    BureaucratReveal,
    /// Discard down to 3 cards in hand.
    MilitiaDiscard,
    /// Trash a Copper for +$3.
    MoneylenderTrash,
    /// Discard one card per empty supply pile.
    PoacherDiscard,
    RemodelTrash,
    RemodelGain,
    /// Choose an Action from hand to play twice.
    ThroneRoomPlay,
    /// Trash one of the revealed Treasures (Bandit, victim's choice).
    BanditTrash,
    /// Set aside a drawn Action instead of keeping it.
    LibrarySetAside,
    MineTrash,
    MineGain,
    /// Trash any of the two cards Sentry revealed.
    SentryTrash,
    /// Discard any of the cards Sentry revealed and did not trash.
    SentryDiscard,
    /// Choose which kept card goes on top of the deck.
    SentryOrder,
    ArtisanGain,
    ArtisanTopdeck,
}

impl Ctx {
    /// Whether the player is choosing during their own turn's main phases.
    pub fn is_main_phase(self) -> bool {
        matches!(self, Ctx::ActionPhase | Ctx::BuyPhase)
    }
}

/// A parked decision: whose it is, what it is about, and what is legal.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Decision {
    pub player: usize,
    pub ctx: Ctx,
    pub options: Vec<Move>,
}

/// Where a gained card goes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dest {
    Discard,
    Hand,
    DeckTop,
}

/// Inline card buffer, so [`Frame`] stays `Copy` and cheap to clone during
/// search. Six slots covers every scratch use in the Base set.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct CardBuf {
    cards: [u8; 6],
    len: u8,
}

impl CardBuf {
    pub const fn new() -> Self {
        CardBuf {
            cards: [0; 6],
            len: 0,
        }
    }
    pub fn push(&mut self, c: Card) {
        assert!((self.len as usize) < 6, "CardBuf overflow");
        self.cards[self.len as usize] = c as u8;
        self.len += 1;
    }
    pub fn remove_one(&mut self, c: Card) -> bool {
        for i in 0..self.len as usize {
            if self.cards[i] == c as u8 {
                for j in i..self.len as usize - 1 {
                    self.cards[j] = self.cards[j + 1];
                }
                self.len -= 1;
                return true;
            }
        }
        false
    }
    pub fn contains(&self, c: Card) -> bool {
        self.iter().any(|x| x == c)
    }
    pub fn len(&self) -> usize {
        self.len as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = Card> + '_ {
        (0..self.len as usize).map(move |i| Card::from_idx(self.cards[i] as usize))
    }
    pub fn get(&self, i: usize) -> Card {
        Card::from_idx(self.cards[i] as usize)
    }
    pub fn from_slice(cards: &[Card]) -> Self {
        let mut b = CardBuf::new();
        for &c in cards {
            b.push(c);
        }
        b
    }
}

/// One entry on the continuation stack.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Frame {
    pub kind: FrameKind,
    /// Player this frame acts on (the victim, for attack frames).
    pub player: u8,
    /// Resumption point within the frame's routine.
    pub step: u8,
    /// General-purpose counter (cards left to draw, coin budget, victim index).
    pub n: u8,
    pub scratch: CardBuf,
}

impl Frame {
    pub fn new(kind: FrameKind, player: usize) -> Self {
        Frame {
            kind,
            player: player as u8,
            step: 0,
            n: 0,
            scratch: CardBuf::new(),
        }
    }
    pub fn with_n(mut self, n: u8) -> Self {
        self.n = n;
        self
    }
    pub fn with_scratch(mut self, s: CardBuf) -> Self {
        self.scratch = s;
        self
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FrameKind {
    // --- turn structure ---
    StartTurn,
    ActionPhase,
    BuyPhase,
    Cleanup,
    // --- generic effects ---
    /// Draw `n` cards for `player`.
    Draw,
    /// Gain `scratch[0]` to destination encoded in `step` (see [`Dest`]).
    Gain,
    /// Resolve the effect text of `scratch[0]`. Used for playing an Action and
    /// for Throne Room's repeat.
    Resolve,
    /// Run the attack `scratch[0]` against each other player in turn order,
    /// `n` = how many victims already handled.
    AttackEach,
    /// Apply attack `scratch[0]` to `player`, after checking for Moat.
    AttackOne,
}

/// A player's zones and per-turn counters.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct PlayerState {
    /// Draw pile. The **last** element is the top of the deck.
    pub deck: Vec<Card>,
    pub hand: Vec<Card>,
    pub discard: Vec<Card>,
    /// Cards played this turn, still in play.
    pub play: Vec<Card>,
    /// Cards set aside by an in-flight effect (Library).
    pub set_aside: Vec<Card>,

    pub actions: u8,
    pub buys: u8,
    pub coins: u8,

    /// Merchants played this turn (each gives +$1 on the first Silver).
    pub merchants: u8,
    pub silver_played: bool,

    pub turns: u32,
}

impl PlayerState {
    pub fn starting(rng: &mut Rng) -> Self {
        let mut deck = vec![Card::Copper; 7];
        deck.extend([Card::Estate; 3]);
        rng.shuffle(&mut deck);
        PlayerState {
            deck,
            ..Default::default()
        }
    }

    /// Every card the player owns, across all zones.
    pub fn all_cards(&self) -> impl Iterator<Item = Card> + '_ {
        self.deck
            .iter()
            .chain(&self.hand)
            .chain(&self.discard)
            .chain(&self.play)
            .chain(&self.set_aside)
            .copied()
    }

    pub fn total_cards(&self) -> usize {
        self.deck.len() + self.hand.len() + self.discard.len() + self.play.len()
            + self.set_aside.len()
    }

    pub fn score(&self) -> i32 {
        let total = self.total_cards() as i32;
        self.all_cards()
            .map(|c| match c {
                Card::Gardens => total / 10,
                other => other.static_vp(),
            })
            .sum()
    }

    pub fn count_owned(&self, card: Card) -> usize {
        self.all_cards().filter(|&c| c == card).count()
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GameState {
    pub players: Vec<PlayerState>,
    /// Remaining copies of each card. Zero for cards not in this kingdom.
    pub supply: [u8; NUM_CARDS],
    /// Which cards are part of this game's supply at all.
    pub in_supply: [bool; NUM_CARDS],
    pub trash: Vec<Card>,

    pub current: u8,
    pub stack: Vec<Frame>,
    pub pending: Option<Decision>,
    /// Move supplied for the pending decision, consumed by the resuming frame.
    pub answer: Option<Move>,

    pub rng: Rng,
    pub over: bool,
    /// Guards against pathological non-terminating games during self-play.
    pub turn_limit: u32,
}

impl GameState {
    #[inline]
    pub fn cur(&self) -> &PlayerState {
        &self.players[self.current as usize]
    }
    #[inline]
    pub fn cur_mut(&mut self) -> &mut PlayerState {
        let i = self.current as usize;
        &mut self.players[i]
    }
    #[inline]
    pub fn num_players(&self) -> usize {
        self.players.len()
    }

    #[inline]
    pub fn supply_of(&self, card: Card) -> u8 {
        self.supply[card.idx()]
    }

    pub fn empty_piles(&self) -> usize {
        (0..NUM_CARDS)
            .filter(|&i| self.in_supply[i] && self.supply[i] == 0)
            .count()
    }

    pub fn game_should_end(&self) -> bool {
        self.supply_of(Card::Province) == 0 || self.empty_piles() >= 3
    }

    /// Cards that can legally be gained for `budget` coins or less.
    pub fn buyable(&self, budget: u8) -> Vec<Card> {
        (0..NUM_CARDS)
            .filter(|&i| self.in_supply[i] && self.supply[i] > 0)
            .map(Card::from_idx)
            .filter(|c| c.cost() <= budget)
            .collect()
    }

    /// Final scores. Ties on VP are broken in favour of whoever took fewer
    /// turns, per the rulebook.
    pub fn scores(&self) -> Vec<i32> {
        self.players.iter().map(|p| p.score()).collect()
    }

    /// Result for each player in `[0.0, 1.0]`: 1 for a win, 0 for a loss, and
    /// an even split among tied players.
    pub fn results(&self) -> Vec<f32> {
        let scores = self.scores();
        let min_turns = self
            .players
            .iter()
            .enumerate()
            .filter(|(i, _)| scores[*i] == *scores.iter().max().unwrap())
            .map(|(_, p)| p.turns)
            .min()
            .unwrap_or(0);
        let best = *scores.iter().max().unwrap();
        let winners: Vec<usize> = (0..self.players.len())
            .filter(|&i| scores[i] == best && self.players[i].turns == min_turns)
            .collect();
        let share = 1.0 / winners.len() as f32;
        (0..self.players.len())
            .map(|i| if winners.contains(&i) { share } else { 0.0 })
            .collect()
    }
}
