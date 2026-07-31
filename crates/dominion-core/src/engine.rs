//! The rules engine.
//!
//! [`Game::advance`] runs the continuation stack until a player has a real
//! choice to make, then parks it in [`GameState::pending`]. Callers answer with
//! [`Game::apply`]. Decisions with only one legal move are resolved
//! automatically, so the caller only ever sees choices that matter — which is
//! also how the online client behaves, and keeps the search tree small.

use crate::card::{Card, ALL_CARDS, KINGDOM_CARDS, NUM_CARDS};
use crate::rng::Rng;
use crate::state::*;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum EngineError {
    NoPendingDecision,
    IllegalMove { got: Move, legal: Vec<Move> },
    BadKingdom(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::NoPendingDecision => write!(f, "no decision is pending"),
            EngineError::IllegalMove { got, legal } => {
                write!(f, "illegal move {got}; legal: {legal:?}")
            }
            EngineError::BadKingdom(m) => write!(f, "bad kingdom: {m}"),
        }
    }
}

impl std::error::Error for EngineError {}

pub struct Game {
    pub state: GameState,
}

/// Distinct cards in a zone, in canonical card order. Selection options are
/// deduplicated: two Coppers in hand offer one `Select(Copper)`, not two.
fn distinct(cards: &[Card]) -> Vec<Card> {
    let mut seen = [false; NUM_CARDS];
    let mut out = Vec::with_capacity(cards.len().min(NUM_CARDS));
    for &c in cards {
        if !seen[c.idx()] {
            seen[c.idx()] = true;
            out.push(c);
        }
    }
    out.sort_unstable();
    out
}

fn remove_one(zone: &mut Vec<Card>, card: Card) -> bool {
    if let Some(i) = zone.iter().position(|&c| c == card) {
        zone.remove(i);
        true
    } else {
        false
    }
}

impl Game {
    /// Start a game. `kingdom` must list exactly 10 distinct Base-set kingdom
    /// cards.
    pub fn new(kingdom: &[Card], num_players: usize, seed: u64) -> Result<Game, EngineError> {
        if kingdom.len() != 10 {
            return Err(EngineError::BadKingdom(format!(
                "expected 10 cards, got {}",
                kingdom.len()
            )));
        }
        if distinct(kingdom).len() != 10 {
            return Err(EngineError::BadKingdom("duplicate cards".into()));
        }
        if let Some(bad) = kingdom.iter().find(|c| !KINGDOM_CARDS.contains(c)) {
            return Err(EngineError::BadKingdom(format!("{bad} is not a kingdom card")));
        }
        if !(2..=MAX_PLAYERS).contains(&num_players) {
            return Err(EngineError::BadKingdom(format!(
                "unsupported player count {num_players}"
            )));
        }

        let mut rng = Rng::new(seed);
        let players: Vec<PlayerState> = (0..num_players)
            .map(|_| PlayerState::starting(&mut rng))
            .collect();

        let victory_pile: u8 = if num_players == 2 { 8 } else { 12 };
        let mut supply = [0u8; NUM_CARDS];
        let mut in_supply = [false; NUM_CARDS];

        for (card, n) in [
            (Card::Copper, 60 - 7 * num_players as u8),
            (Card::Silver, 40),
            (Card::Gold, 30),
            (Card::Estate, victory_pile),
            (Card::Duchy, victory_pile),
            (Card::Province, victory_pile),
            (Card::Curse, 10 * (num_players as u8 - 1)),
        ] {
            supply[card.idx()] = n;
            in_supply[card.idx()] = true;
        }
        for &k in kingdom {
            supply[k.idx()] = if k.is_victory() { victory_pile } else { 10 };
            in_supply[k.idx()] = true;
        }

        let mut state = GameState {
            players,
            supply,
            in_supply,
            trash: Vec::new(),
            current: 0,
            stack: vec![Frame::new(FrameKind::StartTurn, 0)],
            pending: None,
            answer: None,
            rng,
            over: false,
            turn_limit: 250,
        };

        for p in 0..num_players {
            for _ in 0..5 {
                draw_one(&mut state, p);
            }
        }

        let mut game = Game { state };
        game.advance();
        Ok(game)
    }

    /// Convenience: a random legal 10-card kingdom.
    pub fn random_kingdom(rng: &mut Rng) -> Vec<Card> {
        let mut pool = KINGDOM_CARDS.to_vec();
        rng.shuffle(&mut pool);
        pool.truncate(10);
        pool.sort_unstable();
        pool
    }

    #[inline]
    pub fn decision(&self) -> Option<&Decision> {
        self.state.pending.as_ref()
    }

    #[inline]
    pub fn is_over(&self) -> bool {
        self.state.over
    }

    /// Answer the pending decision and run forward to the next one.
    pub fn apply(&mut self, mv: Move) -> Result<(), EngineError> {
        let Some(d) = self.state.pending.take() else {
            return Err(EngineError::NoPendingDecision);
        };
        if !d.options.contains(&mv) {
            let legal = d.options.clone();
            self.state.pending = Some(d);
            return Err(EngineError::IllegalMove { got: mv, legal });
        }
        self.state.answer = Some(mv);
        self.advance();
        Ok(())
    }

    /// Run frames until a decision with more than one option is pending, or the
    /// game ends.
    pub fn advance(&mut self) {
        loop {
            if self.state.over {
                self.state.pending = None;
                return;
            }
            if let Some(d) = &self.state.pending {
                // Forced choices are applied for the caller.
                if d.options.len() == 1 {
                    let mv = d.options[0];
                    self.state.pending = None;
                    self.state.answer = Some(mv);
                } else {
                    return;
                }
            }
            let Some(frame) = self.state.stack.pop() else {
                // Nothing left to do: the game is finished.
                self.state.over = true;
                return;
            };
            self.run(frame);
        }
    }

    fn ask(&mut self, frame: Frame, player: usize, ctx: Ctx, options: Vec<Move>) {
        debug_assert!(!options.is_empty(), "asked {ctx:?} with no options");
        self.state.stack.push(frame);
        self.state.pending = Some(Decision {
            player,
            ctx,
            options,
        });
    }

    fn run(&mut self, mut f: Frame) {
        let ans = self.state.answer.take();
        let p = f.player as usize;
        match f.kind {
            FrameKind::StartTurn => {
                let cur = self.state.current as usize;
                {
                    let pl = &mut self.state.players[cur];
                    pl.actions = 1;
                    pl.buys = 1;
                    pl.coins = 0;
                    pl.merchants = 0;
                    pl.silver_played = false;
                    pl.turns += 1;
                }
                self.state
                    .stack
                    .push(Frame::new(FrameKind::Cleanup, cur));
                self.state
                    .stack
                    .push(Frame::new(FrameKind::BuyPhase, cur));
                self.state
                    .stack
                    .push(Frame::new(FrameKind::ActionPhase, cur));
            }

            FrameKind::ActionPhase => {
                if let Some(mv) = ans {
                    match mv {
                        Move::Done => return,
                        Move::Play(card) => {
                            let pl = &mut self.state.players[p];
                            remove_one(&mut pl.hand, card);
                            pl.play.push(card);
                            pl.actions -= 1;
                            // Continue the phase after the card resolves.
                            self.state.stack.push(Frame::new(FrameKind::ActionPhase, p));
                            self.push_resolve(card, p);
                            return;
                        }
                        _ => unreachable!("action phase answered with {mv}"),
                    }
                }
                let pl = &self.state.players[p];
                if pl.actions == 0 {
                    return;
                }
                let playable: Vec<Card> = distinct(&pl.hand)
                    .into_iter()
                    .filter(|c| c.is_action())
                    .collect();
                if playable.is_empty() {
                    return;
                }
                let mut opts: Vec<Move> = playable.into_iter().map(Move::Play).collect();
                opts.push(Move::Done);
                self.ask(Frame::new(FrameKind::ActionPhase, p), p, Ctx::ActionPhase, opts);
            }

            FrameKind::BuyPhase => {
                if let Some(mv) = ans {
                    match mv {
                        Move::Done => return,
                        Move::Play(card) => {
                            let pl = &mut self.state.players[p];
                            remove_one(&mut pl.hand, card);
                            pl.play.push(card);
                            pl.coins += card.coin_value();
                            if card == Card::Silver && !pl.silver_played {
                                pl.silver_played = true;
                                pl.coins += pl.merchants;
                            }
                        }
                        Move::Buy(card) => {
                            let pl = &mut self.state.players[p];
                            pl.buys -= 1;
                            pl.coins -= card.cost();
                            self.state.stack.push(Frame::new(FrameKind::BuyPhase, p));
                            self.push_gain(card, p, Dest::Discard);
                            return;
                        }
                        Move::Select(_) => unreachable!("buy phase answered with a selection"),
                    }
                    self.state.stack.push(Frame::new(FrameKind::BuyPhase, p));
                    return;
                }
                let pl = &self.state.players[p];
                let mut opts: Vec<Move> = distinct(&pl.hand)
                    .into_iter()
                    .filter(|c| c.is_treasure())
                    .map(Move::Play)
                    .collect();
                if pl.buys > 0 {
                    let budget = pl.coins;
                    opts.extend(self.state.buyable(budget).into_iter().map(Move::Buy));
                }
                opts.push(Move::Done);
                self.ask(Frame::new(FrameKind::BuyPhase, p), p, Ctx::BuyPhase, opts);
            }

            FrameKind::Cleanup => {
                {
                    let pl = &mut self.state.players[p];
                    let played: Vec<Card> = pl.play.drain(..).collect();
                    pl.discard.extend(played);
                    let hand: Vec<Card> = pl.hand.drain(..).collect();
                    pl.discard.extend(hand);
                    let aside: Vec<Card> = pl.set_aside.drain(..).collect();
                    pl.discard.extend(aside);
                }
                for _ in 0..5 {
                    draw_one(&mut self.state, p);
                }
                let over_by_piles = self.state.game_should_end();
                let over_by_limit = self.state.players.iter().all(|pl| pl.turns >= self.state.turn_limit);
                if over_by_piles || over_by_limit {
                    self.state.over = true;
                    return;
                }
                let next = (self.state.current as usize + 1) % self.state.num_players();
                self.state.current = next as u8;
                self.state.stack.push(Frame::new(FrameKind::StartTurn, next));
            }

            FrameKind::Draw => {
                for _ in 0..f.n {
                    if draw_one(&mut self.state, p).is_none() {
                        break;
                    }
                }
            }

            FrameKind::Gain => {
                let card = f.scratch.get(0);
                if self.state.supply[card.idx()] == 0 {
                    return;
                }
                self.state.supply[card.idx()] -= 1;
                let dest = match f.n {
                    1 => Dest::Hand,
                    2 => Dest::DeckTop,
                    _ => Dest::Discard,
                };
                let pl = &mut self.state.players[p];
                match dest {
                    Dest::Discard => pl.discard.push(card),
                    Dest::Hand => pl.hand.push(card),
                    Dest::DeckTop => pl.deck.push(card),
                }
            }

            FrameKind::AttackEach => {
                let attack = f.scratch.get(0);
                let n = self.state.num_players();
                if (f.n as usize) >= n - 1 {
                    return;
                }
                let victim = (self.state.current as usize + 1 + f.n as usize) % n;
                let next = Frame {
                    n: f.n + 1,
                    ..f
                };
                self.state.stack.push(next);
                self.state.stack.push(
                    Frame::new(FrameKind::AttackOne, victim)
                        .with_scratch(CardBuf::from_slice(&[attack])),
                );
            }

            FrameKind::AttackOne => self.run_attack(f, ans),

            FrameKind::Resolve => {
                let card = f.scratch.get(0);
                self.resolve(card, &mut f, ans);
            }
        }
    }

    fn push_resolve(&mut self, card: Card, player: usize) {
        self.state.stack.push(
            Frame::new(FrameKind::Resolve, player).with_scratch(CardBuf::from_slice(&[card])),
        );
    }

    fn push_gain(&mut self, card: Card, player: usize, dest: Dest) {
        let n = match dest {
            Dest::Discard => 0,
            Dest::Hand => 1,
            Dest::DeckTop => 2,
        };
        self.state.stack.push(
            Frame::new(FrameKind::Gain, player)
                .with_n(n)
                .with_scratch(CardBuf::from_slice(&[card])),
        );
    }

    fn push_draw(&mut self, n: u8, player: usize) {
        self.state
            .stack
            .push(Frame::new(FrameKind::Draw, player).with_n(n));
    }

    fn push_attack(&mut self, card: Card, attacker: usize) {
        self.state.stack.push(
            Frame::new(FrameKind::AttackEach, attacker)
                .with_scratch(CardBuf::from_slice(&[card])),
        );
    }

    // ----------------------------------------------------------------- attacks

    fn run_attack(&mut self, mut f: Frame, ans: Option<Move>) {
        let victim = f.player as usize;
        let attack = f.scratch.get(0);

        if f.step == 0 {
            // Moat is the only Reaction in the Base set.
            if self.state.players[victim].hand.contains(&Card::Moat) {
                f.step = 1;
                self.ask(
                    f,
                    victim,
                    Ctx::MoatReveal,
                    vec![Move::Select(Card::Moat), Move::Done],
                );
                return;
            }
            f.step = 2;
        } else if f.step == 1 {
            if ans == Some(Move::Select(Card::Moat)) {
                return; // blocked
            }
            f.step = 2;
        }

        match attack {
            Card::Witch => {
                self.push_gain(Card::Curse, victim, Dest::Discard);
            }

            Card::Militia => {
                if f.step == 2 {
                    if let Some(Move::Select(c)) = ans {
                        remove_one(&mut self.state.players[victim].hand, c);
                        self.state.players[victim].discard.push(c);
                    }
                }
                let hand = &self.state.players[victim].hand;
                if hand.len() <= 3 {
                    return;
                }
                let opts: Vec<Move> = distinct(hand).into_iter().map(Move::Select).collect();
                f.step = 2;
                self.ask(f, victim, Ctx::MilitiaDiscard, opts);
            }

            Card::Bureaucrat => {
                if f.step == 2 {
                    let hand = &self.state.players[victim].hand;
                    let vics: Vec<Card> = distinct(hand)
                        .into_iter()
                        .filter(|c| c.is_victory())
                        .collect();
                    if vics.is_empty() {
                        return; // reveals a hand with no Victory cards
                    }
                    f.step = 3;
                    let opts = vics.into_iter().map(Move::Select).collect();
                    self.ask(f, victim, Ctx::BureaucratReveal, opts);
                } else if let Some(Move::Select(c)) = ans {
                    let pl = &mut self.state.players[victim];
                    remove_one(&mut pl.hand, c);
                    pl.deck.push(c);
                }
            }

            Card::Bandit => {
                if f.step == 2 {
                    let mut revealed = CardBuf::from_slice(&[attack]);
                    for _ in 0..2 {
                        if let Some(c) = reveal_one(&mut self.state, victim) {
                            revealed.push(c);
                        }
                    }
                    f.scratch = revealed;
                    let candidates: Vec<Card> = distinct(
                        &f.scratch.iter().skip(1).collect::<Vec<_>>(),
                    )
                    .into_iter()
                    .filter(|c| c.is_treasure() && *c != Card::Copper)
                    .collect();

                    if candidates.is_empty() {
                        let rest: Vec<Card> = f.scratch.iter().skip(1).collect();
                        self.state.players[victim].discard.extend(rest);
                        return;
                    }
                    f.step = 3;
                    let opts = candidates.into_iter().map(Move::Select).collect();
                    self.ask(f, victim, Ctx::BanditTrash, opts);
                } else {
                    if let Some(Move::Select(c)) = ans {
                        f.scratch.remove_one(c);
                        self.state.trash.push(c);
                    }
                    let rest: Vec<Card> = f.scratch.iter().skip(1).collect();
                    self.state.players[victim].discard.extend(rest);
                }
            }

            other => debug_assert!(false, "{other} is not an attack"),
        }
    }

    // ------------------------------------------------------------ card effects

    fn resolve(&mut self, card: Card, f: &mut Frame, ans: Option<Move>) {
        let p = f.player as usize;
        use Card::*;
        match card {
            // --- vanilla bonuses -------------------------------------------
            Village => {
                self.state.players[p].actions += 2;
                self.push_draw(1, p);
            }
            Smithy => self.push_draw(3, p),
            Moat => self.push_draw(2, p),
            Festival => {
                let pl = &mut self.state.players[p];
                pl.actions += 2;
                pl.buys += 1;
                pl.coins += 2;
            }
            Laboratory => {
                self.state.players[p].actions += 1;
                self.push_draw(2, p);
            }
            Market => {
                let pl = &mut self.state.players[p];
                pl.actions += 1;
                pl.buys += 1;
                pl.coins += 1;
                self.push_draw(1, p);
            }
            Merchant => {
                let pl = &mut self.state.players[p];
                pl.actions += 1;
                pl.merchants += 1;
                self.push_draw(1, p);
            }
            Gardens => {} // never played

            // --- attacks ----------------------------------------------------
            Witch => {
                self.push_attack(Witch, p);
                self.push_draw(2, p);
            }
            Militia => {
                self.state.players[p].coins += 2;
                self.push_attack(Militia, p);
            }
            Bureaucrat => {
                self.push_attack(Bureaucrat, p);
                self.push_gain(Silver, p, Dest::DeckTop);
            }
            Bandit => {
                self.push_attack(Bandit, p);
                self.push_gain(Gold, p, Dest::Discard);
            }

            CouncilRoom => {
                if f.step == 0 {
                    self.state.players[p].buys += 1;
                    f.step = 1;
                    self.state.stack.push(*f);
                    self.push_draw(4, p);
                } else {
                    for other in 0..self.state.num_players() {
                        if other != p {
                            self.push_draw(1, other);
                        }
                    }
                }
            }

            // --- selection effects -----------------------------------------
            Cellar => {
                if f.step == 0 {
                    self.state.players[p].actions += 1;
                    f.step = 1;
                }
                if let Some(mv) = ans {
                    match mv {
                        Move::Select(c) => {
                            let pl = &mut self.state.players[p];
                            remove_one(&mut pl.hand, c);
                            pl.discard.push(c);
                            f.n += 1;
                        }
                        Move::Done => {
                            self.push_draw(f.n, p);
                            return;
                        }
                        _ => unreachable!(),
                    }
                }
                if self.state.players[p].hand.is_empty() {
                    self.push_draw(f.n, p);
                    return;
                }
                let mut opts: Vec<Move> = distinct(&self.state.players[p].hand)
                    .into_iter()
                    .map(Move::Select)
                    .collect();
                opts.push(Move::Done);
                self.ask(*f, p, Ctx::CellarDiscard, opts);
            }

            Chapel => {
                if let Some(mv) = ans {
                    match mv {
                        Move::Select(c) => {
                            remove_one(&mut self.state.players[p].hand, c);
                            self.state.trash.push(c);
                            f.n += 1;
                        }
                        Move::Done => return,
                        _ => unreachable!(),
                    }
                }
                if f.n >= 4 || self.state.players[p].hand.is_empty() {
                    return;
                }
                let mut opts: Vec<Move> = distinct(&self.state.players[p].hand)
                    .into_iter()
                    .map(Move::Select)
                    .collect();
                opts.push(Move::Done);
                self.ask(*f, p, Ctx::ChapelTrash, opts);
            }

            Harbinger => {
                if f.step == 0 {
                    self.state.players[p].actions += 1;
                    f.step = 1;
                    self.state.stack.push(*f);
                    self.push_draw(1, p);
                    return;
                }
                if let Some(Move::Select(c)) = ans {
                    let pl = &mut self.state.players[p];
                    remove_one(&mut pl.discard, c);
                    pl.deck.push(c);
                    return;
                }
                if ans == Some(Move::Done) {
                    return;
                }
                if self.state.players[p].discard.is_empty() {
                    return;
                }
                let mut opts: Vec<Move> = distinct(&self.state.players[p].discard)
                    .into_iter()
                    .map(Move::Select)
                    .collect();
                opts.push(Move::Done);
                self.ask(*f, p, Ctx::HarbingerTopdeck, opts);
            }

            Vassal => {
                if f.step == 0 {
                    self.state.players[p].coins += 2;
                    let Some(top) = reveal_one(&mut self.state, p) else {
                        return;
                    };
                    self.state.players[p].discard.push(top);
                    if !top.is_action() {
                        return;
                    }
                    f.step = 1;
                    self.ask(*f, p, Ctx::VassalPlay, vec![Move::Play(top), Move::Done]);
                } else if let Some(Move::Play(c)) = ans {
                    let pl = &mut self.state.players[p];
                    remove_one(&mut pl.discard, c);
                    pl.play.push(c);
                    self.push_resolve(c, p);
                }
            }

            Workshop => {
                if f.step == 0 {
                    let opts: Vec<Move> =
                        self.state.buyable(4).into_iter().map(Move::Select).collect();
                    if opts.is_empty() {
                        return;
                    }
                    f.step = 1;
                    self.ask(*f, p, Ctx::WorkshopGain, opts);
                } else if let Some(Move::Select(c)) = ans {
                    self.push_gain(c, p, Dest::Discard);
                }
            }

            Moneylender => {
                if f.step == 0 {
                    if !self.state.players[p].hand.contains(&Copper) {
                        return;
                    }
                    f.step = 1;
                    self.ask(
                        *f,
                        p,
                        Ctx::MoneylenderTrash,
                        vec![Move::Select(Copper), Move::Done],
                    );
                } else if ans == Some(Move::Select(Copper)) {
                    remove_one(&mut self.state.players[p].hand, Copper);
                    self.state.trash.push(Copper);
                    self.state.players[p].coins += 3;
                }
            }

            Poacher => {
                match f.step {
                    0 => {
                        let pl = &mut self.state.players[p];
                        pl.actions += 1;
                        pl.coins += 1;
                        f.step = 1;
                        self.state.stack.push(*f);
                        self.push_draw(1, p);
                    }
                    _ => {
                        if f.step == 1 {
                            f.n = self.state.empty_piles() as u8;
                            f.step = 2;
                        }
                        if let Some(Move::Select(c)) = ans {
                            let pl = &mut self.state.players[p];
                            remove_one(&mut pl.hand, c);
                            pl.discard.push(c);
                            f.n -= 1;
                        }
                        if f.n == 0 || self.state.players[p].hand.is_empty() {
                            return;
                        }
                        let opts: Vec<Move> = distinct(&self.state.players[p].hand)
                            .into_iter()
                            .map(Move::Select)
                            .collect();
                        self.ask(*f, p, Ctx::PoacherDiscard, opts);
                    }
                }
            }

            Remodel => {
                match f.step {
                    0 => {
                        if self.state.players[p].hand.is_empty() {
                            return;
                        }
                        f.step = 1;
                        let opts: Vec<Move> = distinct(&self.state.players[p].hand)
                            .into_iter()
                            .map(Move::Select)
                            .collect();
                        self.ask(*f, p, Ctx::RemodelTrash, opts);
                    }
                    1 => {
                        let Some(Move::Select(c)) = ans else { return };
                        remove_one(&mut self.state.players[p].hand, c);
                        self.state.trash.push(c);
                        let opts: Vec<Move> = self
                            .state
                            .buyable(c.cost() + 2)
                            .into_iter()
                            .map(Move::Select)
                            .collect();
                        if opts.is_empty() {
                            return;
                        }
                        f.step = 2;
                        self.ask(*f, p, Ctx::RemodelGain, opts);
                    }
                    _ => {
                        if let Some(Move::Select(c)) = ans {
                            self.push_gain(c, p, Dest::Discard);
                        }
                    }
                }
            }

            Mine => {
                match f.step {
                    0 => {
                        let treasures: Vec<Card> = distinct(&self.state.players[p].hand)
                            .into_iter()
                            .filter(|c| c.is_treasure())
                            .collect();
                        if treasures.is_empty() {
                            return;
                        }
                        f.step = 1;
                        let mut opts: Vec<Move> =
                            treasures.into_iter().map(Move::Select).collect();
                        opts.push(Move::Done);
                        self.ask(*f, p, Ctx::MineTrash, opts);
                    }
                    1 => {
                        let Some(Move::Select(c)) = ans else { return };
                        remove_one(&mut self.state.players[p].hand, c);
                        self.state.trash.push(c);
                        let opts: Vec<Move> = self
                            .state
                            .buyable(c.cost() + 3)
                            .into_iter()
                            .filter(|g| g.is_treasure())
                            .map(Move::Select)
                            .collect();
                        if opts.is_empty() {
                            return;
                        }
                        f.step = 2;
                        self.ask(*f, p, Ctx::MineGain, opts);
                    }
                    _ => {
                        if let Some(Move::Select(c)) = ans {
                            self.push_gain(c, p, Dest::Hand);
                        }
                    }
                }
            }

            ThroneRoom => {
                if f.step == 0 {
                    let actions: Vec<Card> = distinct(&self.state.players[p].hand)
                        .into_iter()
                        .filter(|c| c.is_action())
                        .collect();
                    if actions.is_empty() {
                        return;
                    }
                    f.step = 1;
                    let mut opts: Vec<Move> = actions.into_iter().map(Move::Play).collect();
                    opts.push(Move::Done);
                    self.ask(*f, p, Ctx::ThroneRoomPlay, opts);
                } else if let Some(Move::Play(c)) = ans {
                    let pl = &mut self.state.players[p];
                    remove_one(&mut pl.hand, c);
                    pl.play.push(c);
                    self.push_resolve(c, p);
                    self.push_resolve(c, p);
                }
            }

            Library => {
                if f.step == 0 {
                    f.step = 1;
                } else if let Some(Move::Select(c)) = ans {
                    // Set the just-drawn Action aside instead of keeping it.
                    let pl = &mut self.state.players[p];
                    remove_one(&mut pl.hand, c);
                    pl.set_aside.push(c);
                }
                loop {
                    if self.state.players[p].hand.len() >= 7 {
                        break;
                    }
                    let Some(drawn) = draw_one(&mut self.state, p) else {
                        break;
                    };
                    if drawn.is_action() {
                        self.ask(
                            *f,
                            p,
                            Ctx::LibrarySetAside,
                            vec![Move::Select(drawn), Move::Done],
                        );
                        return;
                    }
                }
                let pl = &mut self.state.players[p];
                let aside: Vec<Card> = pl.set_aside.drain(..).collect();
                pl.discard.extend(aside);
            }

            Sentry => {
                match f.step {
                    0 => {
                        self.state.players[p].actions += 1;
                        f.step = 1;
                        self.state.stack.push(*f);
                        self.push_draw(1, p);
                    }
                    1 => {
                        // Look at the top two cards; they live in scratch[1..].
                        for _ in 0..2 {
                            if let Some(c) = reveal_one(&mut self.state, p) {
                                f.scratch.push(c);
                            }
                        }
                        f.step = 2;
                        self.sentry_trash(f);
                    }
                    2 => {
                        match ans {
                            Some(Move::Select(c)) => {
                                f.scratch.remove_one(c);
                                self.state.trash.push(c);
                                self.sentry_trash(f);
                            }
                            _ => {
                                f.step = 3;
                                self.sentry_discard(f);
                            }
                        }
                    }
                    3 => match ans {
                        Some(Move::Select(c)) => {
                            f.scratch.remove_one(c);
                            self.state.players[p].discard.push(c);
                            self.sentry_discard(f);
                        }
                        _ => {
                            f.step = 4;
                            self.sentry_order(f, None);
                        }
                    },
                    _ => self.sentry_order(f, ans),
                }
            }

            Artisan => {
                match f.step {
                    0 => {
                        let opts: Vec<Move> =
                            self.state.buyable(5).into_iter().map(Move::Select).collect();
                        if opts.is_empty() {
                            f.step = 2;
                            self.state.stack.push(*f);
                            return;
                        }
                        f.step = 1;
                        self.ask(*f, p, Ctx::ArtisanGain, opts);
                    }
                    1 => {
                        if let Some(Move::Select(c)) = ans {
                            f.step = 2;
                            self.state.stack.push(*f);
                            self.push_gain(c, p, Dest::Hand);
                        }
                    }
                    2 => {
                        if self.state.players[p].hand.is_empty() {
                            return;
                        }
                        f.step = 3;
                        let opts: Vec<Move> = distinct(&self.state.players[p].hand)
                            .into_iter()
                            .map(Move::Select)
                            .collect();
                        self.ask(*f, p, Ctx::ArtisanTopdeck, opts);
                    }
                    _ => {
                        if let Some(Move::Select(c)) = ans {
                            let pl = &mut self.state.players[p];
                            remove_one(&mut pl.hand, c);
                            pl.deck.push(c);
                        }
                    }
                }
            }

            // Treasures and Victory cards have no on-play Action effect.
            Copper | Silver | Gold | Estate | Duchy | Province | Curse => {}
        }
    }

    fn sentry_trash(&mut self, f: &mut Frame) {
        let p = f.player as usize;
        let looked: Vec<Card> = f.scratch.iter().skip(1).collect();
        if looked.is_empty() {
            return;
        }
        let mut opts: Vec<Move> = distinct(&looked).into_iter().map(Move::Select).collect();
        opts.push(Move::Done);
        f.step = 2;
        self.ask(*f, p, Ctx::SentryTrash, opts);
    }

    fn sentry_discard(&mut self, f: &mut Frame) {
        let p = f.player as usize;
        let looked: Vec<Card> = f.scratch.iter().skip(1).collect();
        if looked.is_empty() {
            return;
        }
        let mut opts: Vec<Move> = distinct(&looked).into_iter().map(Move::Select).collect();
        opts.push(Move::Done);
        f.step = 3;
        self.ask(*f, p, Ctx::SentryDiscard, opts);
    }

    fn sentry_order(&mut self, f: &mut Frame, ans: Option<Move>) {
        let p = f.player as usize;
        let looked: Vec<Card> = f.scratch.iter().skip(1).collect();
        if let Some(Move::Select(top)) = ans {
            let mut rest = looked.clone();
            remove_one(&mut rest, top);
            let pl = &mut self.state.players[p];
            pl.deck.extend(rest);
            pl.deck.push(top);
            return;
        }
        match looked.len() {
            0 => {}
            1 => self.state.players[p].deck.push(looked[0]),
            _ => {
                let opts = distinct(&looked);
                if opts.len() == 1 {
                    self.state.players[p].deck.extend(looked);
                } else {
                    f.step = 4;
                    let opts: Vec<Move> = opts.into_iter().map(Move::Select).collect();
                    self.ask(*f, p, Ctx::SentryOrder, opts);
                }
            }
        }
    }
}

/// Move the top card of `p`'s deck into hand, reshuffling the discard pile if
/// the deck has run out. Returns `None` only when the player owns no more
/// cards to draw.
pub fn draw_one(state: &mut GameState, p: usize) -> Option<Card> {
    let card = pop_deck(state, p)?;
    state.players[p].hand.push(card);
    Some(card)
}

/// Like [`draw_one`] but leaves the card in the caller's hands (Vassal,
/// Sentry, Bandit) rather than putting it into hand.
pub fn reveal_one(state: &mut GameState, p: usize) -> Option<Card> {
    pop_deck(state, p)
}

fn pop_deck(state: &mut GameState, p: usize) -> Option<Card> {
    if state.players[p].deck.is_empty() {
        if state.players[p].discard.is_empty() {
            return None;
        }
        let pl = &mut state.players[p];
        std::mem::swap(&mut pl.deck, &mut pl.discard);
        let deck = std::mem::take(&mut pl.deck);
        let mut deck = deck;
        state.rng.shuffle(&mut deck);
        state.players[p].deck = deck;
    }
    state.players[p].deck.pop()
}

/// All cards that could appear in any game, for feature encoding.
pub fn all_cards() -> &'static [Card; NUM_CARDS] {
    &ALL_CARDS
}
