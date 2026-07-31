//! Targeted rule tests.
//!
//! The fuzzer catches crashes and stuck games; these catch silently wrong
//! rules, which would otherwise teach the AI a game that isn't Dominion.

use dominion_core::state::{Frame, FrameKind};
use dominion_core::{Card::*, Ctx, Game, GameState, Move, KINGDOM_CARDS};
use dominion_core::Card;

/// A 10-card kingdom containing `wanted`, padded deterministically.
fn king(wanted: &[Card]) -> Vec<Card> {
    let mut k: Vec<Card> = wanted.to_vec();
    for &c in KINGDOM_CARDS.iter() {
        if k.len() == 10 {
            break;
        }
        if !k.contains(&c) {
            k.push(c);
        }
    }
    k.sort_unstable();
    k
}

/// Build a game, overwrite the state with a hand-crafted scenario, and run the
/// current player's action and buy phases over it.
fn scenario(wanted: &[Card], setup: impl FnOnce(&mut GameState)) -> Game {
    let mut g = Game::new(&king(wanted), 2, 1).unwrap();
    g.state.pending = None;
    g.state.answer = None;
    g.state.stack.clear();
    for p in &mut g.state.players {
        p.hand.clear();
        p.deck.clear();
        p.discard.clear();
        p.play.clear();
    }
    let cur = g.state.current as usize;
    {
        let pl = &mut g.state.players[cur];
        pl.actions = 1;
        pl.buys = 1;
        pl.coins = 0;
    }
    setup(&mut g.state);
    g.state.stack.push(Frame::new(FrameKind::BuyPhase, cur));
    g.state.stack.push(Frame::new(FrameKind::ActionPhase, cur));
    g.advance();
    g
}

#[track_caller]
fn play(g: &mut Game, mv: Move) {
    let d = g.decision().unwrap_or_else(|| panic!("no decision pending, wanted {mv}"));
    assert!(
        d.options.contains(&mv),
        "{mv} not offered in {:?}; options: {:?}",
        d.ctx,
        d.options
    );
    g.apply(mv).unwrap();
}

#[track_caller]
fn expect_ctx(g: &Game, ctx: Ctx) {
    let d = g.decision().unwrap_or_else(|| panic!("no decision pending, wanted {ctx:?}"));
    assert_eq!(d.ctx, ctx, "options were {:?}", d.options);
}

fn hand_of(g: &Game, p: usize) -> Vec<Card> {
    let mut h = g.state.players[p].hand.clone();
    h.sort_unstable();
    h
}

fn count(zone: &[Card], c: Card) -> usize {
    zone.iter().filter(|&&x| x == c).count()
}

// ------------------------------------------------------------ card data

/// Costs are the foundation every buy decision rests on, and a wrong one is
/// invisible at runtime — it just quietly makes the bots play a different game.
#[test]
fn every_card_costs_what_it_says_on_the_box() {
    let expected: &[(Card, u8)] = &[
        (Copper, 0),
        (Curse, 0),
        (Estate, 2),
        (Silver, 3),
        (Duchy, 5),
        (Gold, 6),
        (Province, 8),
        (Cellar, 2),
        (Chapel, 2),
        (Moat, 2),
        (Harbinger, 3),
        (Merchant, 3),
        (Vassal, 3),
        (Village, 3),
        (Workshop, 3),
        (Bureaucrat, 4),
        (Gardens, 4),
        (Militia, 4),
        (Moneylender, 4),
        (Poacher, 4),
        (Remodel, 4),
        (Smithy, 4),
        (ThroneRoom, 4),
        (Bandit, 5),
        (CouncilRoom, 5),
        (Festival, 5),
        (Laboratory, 5),
        (Library, 5),
        (Market, 5),
        (Mine, 5),
        (Sentry, 5),
        (Witch, 5),
        (Artisan, 6),
    ];
    assert_eq!(expected.len(), dominion_core::NUM_CARDS, "a card is missing");
    for &(card, cost) in expected {
        assert_eq!(card.cost(), cost, "{card} should cost ${cost}");
    }
}

#[test]
fn card_types_are_right() {
    assert!(Moat.is_action() && Moat.is_reaction());
    assert!(Gardens.is_victory() && !Gardens.is_action());
    for c in [Militia, Witch, Bandit, Bureaucrat] {
        assert!(c.is_attack() && c.is_action(), "{c} is an Attack");
    }
    assert_eq!(Copper.coin_value(), 1);
    assert_eq!(Silver.coin_value(), 2);
    assert_eq!(Gold.coin_value(), 3);
}

// ---------------------------------------------------------------- vanilla

#[test]
fn village_draws_one_and_gives_two_actions() {
    let mut g = scenario(&[Village], |s| {
        s.players[0].hand = vec![Village, Village, Estate];
        s.players[0].deck = vec![Copper; 5];
    });
    play(&mut g, Move::Play(Village));
    // +1 Card, +2 Actions: one action was spent playing it, so 2 remain.
    assert_eq!(g.state.players[0].actions, 2);
    assert_eq!(g.state.players[0].hand.len(), 3);
}

/// Merchant's bonus is per Merchant played, and only on the *first* Silver.
#[test]
fn merchant_bonus_stacks_and_fires_once() {
    let mut g = scenario(&[Merchant], |s| {
        s.players[0].hand = vec![Merchant, Merchant, Silver, Silver];
        s.players[0].deck = vec![Copper; 5];
    });
    play(&mut g, Move::Play(Merchant));
    play(&mut g, Move::Play(Merchant));
    // No Actions left in hand, so the action phase ends by itself.
    expect_ctx(&g, Ctx::BuyPhase);
    play(&mut g, Move::Play(Silver));
    assert_eq!(g.state.players[0].coins, 4, "$2 from Silver + $1 per Merchant");
    play(&mut g, Move::Play(Silver));
    assert_eq!(g.state.players[0].coins, 6, "second Silver gets no bonus");
}

#[test]
fn throne_room_resolves_the_card_twice() {
    let mut g = scenario(&[ThroneRoom, Smithy], |s| {
        s.players[0].hand = vec![ThroneRoom, Smithy];
        s.players[0].deck = vec![Copper; 10];
    });
    play(&mut g, Move::Play(ThroneRoom));
    expect_ctx(&g, Ctx::ThroneRoomPlay);
    play(&mut g, Move::Play(Smithy));
    assert_eq!(g.state.players[0].hand.len(), 6, "3 cards, twice");
}

// ---------------------------------------------------------------- attacks

#[test]
fn moat_blocks_militia() {
    let mut g = scenario(&[Militia, Moat], |s| {
        s.players[0].hand = vec![Militia];
        s.players[0].deck = vec![Copper; 5];
        s.players[1].hand = vec![Moat, Copper, Copper, Copper, Estate];
    });
    play(&mut g, Move::Play(Militia));
    expect_ctx(&g, Ctx::MoatReveal);
    play(&mut g, Move::Select(Moat));
    assert_eq!(g.state.players[1].hand.len(), 5, "attack was blocked");
    assert_eq!(g.state.players[0].coins, 2, "Militia still gives +$2");
}

#[test]
fn militia_forces_a_discard_to_three() {
    let mut g = scenario(&[Militia], |s| {
        s.players[0].hand = vec![Militia];
        s.players[0].deck = vec![Copper; 5];
        s.players[1].hand = vec![Copper, Copper, Estate, Estate, Gold];
    });
    play(&mut g, Move::Play(Militia));
    expect_ctx(&g, Ctx::MilitiaDiscard);
    play(&mut g, Move::Select(Estate));
    expect_ctx(&g, Ctx::MilitiaDiscard);
    play(&mut g, Move::Select(Estate));
    assert_eq!(hand_of(&g, 1), vec![Copper, Copper, Gold]);
    assert_eq!(count(&g.state.players[1].discard, Estate), 2);
}

#[test]
fn witch_hands_out_a_curse() {
    let mut g = scenario(&[Witch], |s| {
        s.players[0].hand = vec![Witch];
        s.players[0].deck = vec![Copper; 5];
    });
    let curses_before = g.state.supply_of(Curse);
    play(&mut g, Move::Play(Witch));
    assert_eq!(count(&g.state.players[1].discard, Curse), 1);
    assert_eq!(g.state.supply_of(Curse), curses_before - 1);
    assert_eq!(g.state.players[0].hand.len(), 2, "+2 Cards");
}

#[test]
fn bureaucrat_topdecks_a_victory_card() {
    let mut g = scenario(&[Bureaucrat], |s| {
        s.players[0].hand = vec![Bureaucrat];
        s.players[0].deck = vec![Copper; 5];
        s.players[1].hand = vec![Copper, Duchy, Estate];
        s.players[1].deck = vec![Copper; 3];
    });
    play(&mut g, Move::Play(Bureaucrat));
    assert_eq!(
        *g.state.players[0].deck.last().unwrap(),
        Silver,
        "attacker gains a Silver onto their deck"
    );
    expect_ctx(&g, Ctx::BureaucratReveal);
    play(&mut g, Move::Select(Estate));
    assert_eq!(*g.state.players[1].deck.last().unwrap(), Estate);
    assert_eq!(hand_of(&g, 1), vec![Copper, Duchy]);
}

#[test]
fn bandit_trashes_a_non_copper_treasure() {
    let mut g = scenario(&[Bandit], |s| {
        s.players[0].hand = vec![Bandit];
        s.players[0].deck = vec![Copper; 5];
        // Top of deck is the last element: Gold, then Copper.
        s.players[1].deck = vec![Estate, Copper, Gold];
    });
    play(&mut g, Move::Play(Bandit));
    // Gold and Copper revealed; only Gold is a legal trash target, so the
    // engine resolves it without asking.
    assert_eq!(count(&g.state.trash, Gold), 1);
    assert_eq!(count(&g.state.players[1].discard, Copper), 1);
    assert_eq!(count(&g.state.players[0].discard, Gold), 1, "attacker gains a Gold");
}

#[test]
fn bandit_lets_the_victim_choose_between_treasures() {
    let mut g = scenario(&[Bandit], |s| {
        s.players[0].hand = vec![Bandit];
        s.players[0].deck = vec![Copper; 5];
        s.players[1].deck = vec![Silver, Gold];
    });
    play(&mut g, Move::Play(Bandit));
    expect_ctx(&g, Ctx::BanditTrash);
    assert_eq!(g.decision().unwrap().player, 1, "the victim chooses");
    play(&mut g, Move::Select(Silver));
    assert_eq!(count(&g.state.trash, Silver), 1);
    assert_eq!(count(&g.state.players[1].discard, Gold), 1);
}

// ------------------------------------------------------------- selection

#[test]
fn cellar_discards_then_draws_the_same_number() {
    let mut g = scenario(&[Cellar], |s| {
        s.players[0].hand = vec![Cellar, Estate, Estate, Copper];
        s.players[0].deck = vec![Gold; 5];
    });
    play(&mut g, Move::Play(Cellar));
    expect_ctx(&g, Ctx::CellarDiscard);
    play(&mut g, Move::Select(Estate));
    play(&mut g, Move::Select(Estate));
    play(&mut g, Move::Done);
    assert_eq!(hand_of(&g, 0), vec![Copper, Gold, Gold]);
    assert_eq!(g.state.players[0].actions, 1, "+1 Action");
}

#[test]
fn chapel_trashes_at_most_four() {
    let mut g = scenario(&[Chapel], |s| {
        s.players[0].hand = vec![Chapel, Copper, Copper, Copper, Copper, Copper];
        s.players[0].deck = vec![Gold; 5];
    });
    play(&mut g, Move::Play(Chapel));
    for _ in 0..4 {
        expect_ctx(&g, Ctx::ChapelTrash);
        play(&mut g, Move::Select(Copper));
    }
    assert_eq!(count(&g.state.trash, Copper), 4);
    assert_eq!(hand_of(&g, 0), vec![Copper], "the fifth Copper survives");
}

#[test]
fn vassal_may_play_the_discarded_action() {
    let mut g = scenario(&[Vassal, Smithy], |s| {
        s.players[0].hand = vec![Vassal];
        s.players[0].deck = vec![Copper, Copper, Copper, Copper, Smithy];
    });
    play(&mut g, Move::Play(Vassal));
    expect_ctx(&g, Ctx::VassalPlay);
    play(&mut g, Move::Play(Smithy));
    assert_eq!(g.state.players[0].coins, 2, "+$2");
    assert_eq!(g.state.players[0].hand.len(), 3, "Smithy drew 3");
    assert!(g.state.players[0].play.contains(&Smithy));
    assert_eq!(g.state.players[0].actions, 0, "Vassal's play is free but the Vassal cost one");
}

#[test]
fn harbinger_pulls_a_card_out_of_the_discard_pile() {
    let mut g = scenario(&[Harbinger], |s| {
        s.players[0].hand = vec![Harbinger];
        s.players[0].deck = vec![Copper; 5];
        s.players[0].discard = vec![Estate, Gold];
    });
    play(&mut g, Move::Play(Harbinger));
    expect_ctx(&g, Ctx::HarbingerTopdeck);
    play(&mut g, Move::Select(Gold));
    assert_eq!(*g.state.players[0].deck.last().unwrap(), Gold);
    assert_eq!(g.state.players[0].discard, vec![Estate]);
}

#[test]
fn mine_upgrades_a_treasure_into_hand() {
    let mut g = scenario(&[Mine], |s| {
        s.players[0].hand = vec![Mine, Copper];
        s.players[0].deck = vec![Estate; 5];
    });
    play(&mut g, Move::Play(Mine));
    expect_ctx(&g, Ctx::MineTrash);
    play(&mut g, Move::Select(Copper));
    expect_ctx(&g, Ctx::MineGain);
    let opts = &g.decision().unwrap().options;
    assert!(opts.contains(&Move::Select(Silver)));
    assert!(!opts.contains(&Move::Select(Gold)), "Gold costs $5, budget is $3");
    play(&mut g, Move::Select(Silver));
    assert_eq!(hand_of(&g, 0), vec![Silver], "gained to hand, not discard");
    assert_eq!(count(&g.state.trash, Copper), 1);
}

#[test]
fn remodel_gains_two_more_than_the_trashed_card() {
    let mut g = scenario(&[Remodel], |s| {
        s.players[0].hand = vec![Remodel, Estate];
        s.players[0].deck = vec![Copper; 5];
    });
    play(&mut g, Move::Play(Remodel));
    // Estate is the only card in hand, so the trash choice is forced.
    expect_ctx(&g, Ctx::RemodelGain);
    let opts = &g.decision().unwrap().options;
    assert!(opts.contains(&Move::Select(Silver)), "$2 + 2 = $4");
    assert!(!opts.contains(&Move::Select(Gold)));
    play(&mut g, Move::Select(Silver));
    assert_eq!(count(&g.state.players[0].discard, Silver), 1);
}

#[test]
fn library_skips_chosen_actions_and_stops_at_seven() {
    let mut g = scenario(&[Library, Village], |s| {
        s.players[0].hand = vec![Library];
        let mut deck = vec![Copper; 7];
        deck.push(Village); // top of deck
        s.players[0].deck = deck;
    });
    play(&mut g, Move::Play(Library));
    expect_ctx(&g, Ctx::LibrarySetAside);
    play(&mut g, Move::Select(Village));
    assert_eq!(g.state.players[0].hand.len(), 7);
    assert!(g.state.players[0].hand.iter().all(|&c| c == Copper));
    assert_eq!(
        count(&g.state.players[0].discard, Village),
        1,
        "set-aside cards are discarded afterwards"
    );
    assert!(g.state.players[0].set_aside.is_empty());
}

#[test]
fn library_keeps_an_action_when_asked_to() {
    let mut g = scenario(&[Library, Village], |s| {
        s.players[0].hand = vec![Library];
        let mut deck = vec![Copper; 7];
        deck.push(Village);
        s.players[0].deck = deck;
    });
    play(&mut g, Move::Play(Library));
    play(&mut g, Move::Done); // keep the Village
    assert_eq!(g.state.players[0].hand.len(), 7);
    assert_eq!(count(&g.state.players[0].hand, Village), 1);
}

#[test]
fn sentry_can_trash_one_card_and_discard_another() {
    let mut g = scenario(&[Sentry], |s| {
        s.players[0].hand = vec![Sentry];
        // Top of deck last: Copper is drawn by +1 Card, then Curse and Estate
        // are looked at.
        s.players[0].deck = vec![Gold, Estate, Curse, Copper];
    });
    play(&mut g, Move::Play(Sentry));
    assert_eq!(hand_of(&g, 0), vec![Copper], "+1 Card");
    expect_ctx(&g, Ctx::SentryTrash);
    play(&mut g, Move::Select(Curse));
    expect_ctx(&g, Ctx::SentryTrash);
    play(&mut g, Move::Done);
    expect_ctx(&g, Ctx::SentryDiscard);
    play(&mut g, Move::Select(Estate));
    assert_eq!(count(&g.state.trash, Curse), 1);
    assert_eq!(count(&g.state.players[0].discard, Estate), 1);
    assert_eq!(g.state.players[0].deck, vec![Gold], "the rest stays on the deck");
}

#[test]
fn sentry_puts_kept_cards_back_in_the_chosen_order() {
    let mut g = scenario(&[Sentry], |s| {
        s.players[0].hand = vec![Sentry];
        s.players[0].deck = vec![Estate, Gold, Copper];
    });
    play(&mut g, Move::Play(Sentry));
    play(&mut g, Move::Done); // trash nothing
    play(&mut g, Move::Done); // discard nothing
    expect_ctx(&g, Ctx::SentryOrder);
    play(&mut g, Move::Select(Estate)); // Estate on top
    assert_eq!(g.state.players[0].deck, vec![Gold, Estate]);
}

#[test]
fn artisan_gains_to_hand_then_topdecks() {
    let mut g = scenario(&[Artisan, Market], |s| {
        s.players[0].hand = vec![Artisan, Copper];
        s.players[0].deck = vec![Estate; 3];
    });
    play(&mut g, Move::Play(Artisan));
    expect_ctx(&g, Ctx::ArtisanGain);
    let opts = &g.decision().unwrap().options;
    assert!(opts.contains(&Move::Select(Market)), "$5 budget reaches Market");
    assert!(!opts.contains(&Move::Select(Gold)), "Gold costs $6");
    play(&mut g, Move::Select(Market));
    expect_ctx(&g, Ctx::ArtisanTopdeck);
    play(&mut g, Move::Select(Market));
    assert_eq!(*g.state.players[0].deck.last().unwrap(), Market);
    assert_eq!(hand_of(&g, 0), vec![Copper]);
}

#[test]
fn poacher_discards_one_card_per_empty_pile() {
    let mut g = scenario(&[Poacher], |s| {
        s.players[0].hand = vec![Poacher, Copper, Estate, Estate];
        s.players[0].deck = vec![Gold; 3];
        s.supply[Cellar.idx()] = 0;
        s.supply[Chapel.idx()] = 0;
    });
    play(&mut g, Move::Play(Poacher));
    assert_eq!(g.state.players[0].coins, 1);
    expect_ctx(&g, Ctx::PoacherDiscard);
    play(&mut g, Move::Select(Estate));
    expect_ctx(&g, Ctx::PoacherDiscard);
    play(&mut g, Move::Select(Estate));
    assert_eq!(hand_of(&g, 0), vec![Copper, Gold]);
}

#[test]
fn moneylender_trashes_a_copper_for_three() {
    let mut g = scenario(&[Moneylender], |s| {
        s.players[0].hand = vec![Moneylender, Copper, Estate];
        s.players[0].deck = vec![Gold; 3];
    });
    play(&mut g, Move::Play(Moneylender));
    expect_ctx(&g, Ctx::MoneylenderTrash);
    play(&mut g, Move::Select(Copper));
    assert_eq!(g.state.players[0].coins, 3);
    assert_eq!(count(&g.state.trash, Copper), 1);
}

#[test]
fn council_room_draws_for_everyone_and_adds_a_buy() {
    let mut g = scenario(&[CouncilRoom], |s| {
        s.players[0].hand = vec![CouncilRoom];
        s.players[0].deck = vec![Copper; 8];
        s.players[1].deck = vec![Estate; 3];
    });
    play(&mut g, Move::Play(CouncilRoom));
    assert_eq!(g.state.players[0].hand.len(), 4);
    assert_eq!(g.state.players[0].buys, 2);
    assert_eq!(g.state.players[1].hand.len(), 1, "each other player draws one");
}

#[test]
fn workshop_gains_up_to_four() {
    let mut g = scenario(&[Workshop], |s| {
        s.players[0].hand = vec![Workshop];
        s.players[0].deck = vec![Copper; 3];
    });
    play(&mut g, Move::Play(Workshop));
    expect_ctx(&g, Ctx::WorkshopGain);
    let opts = &g.decision().unwrap().options;
    assert!(opts.contains(&Move::Select(Silver)));
    assert!(!opts.contains(&Move::Select(Gold)));
    play(&mut g, Move::Select(Silver));
    assert_eq!(count(&g.state.players[0].discard, Silver), 1);
}

// -------------------------------------------------------------- scoring

#[test]
fn gardens_scores_one_per_ten_cards() {
    let mut g = Game::new(&king(&[Gardens]), 2, 3).unwrap();
    let p = &mut g.state.players[0];
    p.deck.clear();
    p.hand.clear();
    p.discard.clear();
    p.play.clear();
    p.set_aside.clear();
    // 24 cards, two of them Gardens: 24/10 = 2 VP each.
    p.deck = vec![Copper; 22];
    p.deck.push(Gardens);
    p.deck.push(Gardens);
    assert_eq!(p.total_cards(), 24);
    assert_eq!(p.score(), 4);

    g.state.players[0].deck.push(Copper); // 25 cards, still 2 each
    assert_eq!(g.state.players[0].score(), 4);
    for _ in 0..5 {
        g.state.players[0].deck.push(Copper); // 30 cards -> 3 each
    }
    assert_eq!(g.state.players[0].score(), 6);
}

#[test]
fn curses_and_provinces_score_correctly() {
    let mut g = Game::new(&king(&[Cellar]), 2, 3).unwrap();
    let p = &mut g.state.players[0];
    p.deck.clear();
    p.hand.clear();
    p.discard.clear();
    p.play.clear();
    p.deck = vec![Province, Province, Duchy, Estate, Curse, Curse];
    assert_eq!(p.score(), 6 + 6 + 3 + 1 - 1 - 1);
}

#[test]
fn ties_are_broken_by_fewer_turns() {
    let mut g = Game::new(&king(&[Cellar]), 2, 3).unwrap();
    for p in g.state.players.iter_mut() {
        p.deck.clear();
        p.hand.clear();
        p.discard.clear();
        p.play.clear();
        p.deck = vec![Province];
    }
    g.state.players[0].turns = 10;
    g.state.players[1].turns = 9;
    assert_eq!(g.state.results(), vec![0.0, 1.0]);

    g.state.players[1].turns = 10;
    assert_eq!(g.state.results(), vec![0.5, 0.5], "equal turns is a real tie");
}
