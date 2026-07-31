use dominion_ai::{search, MctsConfig};
use dominion_bots::buy::{big_money_smithy, MenuBot};
use dominion_bots::Agent;
use dominion_core::{Card, Game, Rng};

fn main() {
    let cfg = MctsConfig { worlds: 4, iterations: 200, ..Default::default() };
    let mut rng = Rng::new(3);
    let kingdom = vec![Card::Smithy, Card::Village, Card::Market, Card::Militia, Card::Cellar,
                       Card::Moat, Card::Chapel, Card::Mine, Card::Remodel, Card::Workshop];
    let mut opp = MenuBot::new(big_money_smithy());
    let mut g = Game::new(&kingdom, 2, 7).unwrap();
    let mut shown = 0;
    while !g.is_over() && shown < 14 {
        let d = g.decision().unwrap().clone();
        let mv = if d.player == 0 {
            let (best, stats) = search(&g.state, &d, &cfg, &mut rng);
            if d.ctx == dominion_core::Ctx::BuyPhase {
                let mut s: Vec<_> = stats.iter().filter(|(_, v)| *v > 0).collect();
                s.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
                println!("T{} coins={} -> {}  {:?}", g.state.players[0].turns,
                    g.state.players[0].coins, best,
                    s.iter().take(5).map(|(m, v)| format!("{m}:{v}")).collect::<Vec<_>>());
                shown += 1;
            }
            best
        } else {
            opp.decide(&g.state, &d)
        };
        g.apply(mv).unwrap();
    }
}
