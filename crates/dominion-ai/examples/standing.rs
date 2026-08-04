//! Where the AI actually stands, in one command.
//!
//! Individual experiments here compare a change against its own control, which
//! answers "did this help" but never "how strong is the thing". This measures
//! the shipping configuration — the default `MctsConfig`, which now prices
//! leaves by rollout — against the three opponents whose strength is known:
//!
//! * the heuristic policy the search uses as its own prior and rollout, so
//!   anything above 50% is strength the search added rather than a better
//!   hand-written strategy;
//! * the strongest hand-written buy menu (Double Witch); and
//! * the same search without a network, which isolates what training bought.
//!
//! Every number is reported with its standard error, because the recurring
//! mistake in this project has been acting on a point estimate that later
//! moved by 2σ when re-run at a larger sample.

use dominion_ai::mcts::NetMctsAgent;
use dominion_ai::{MctsAgent, MctsConfig, Net};
use dominion_bots::buy::{double_witch, required_kingdom, MenuBot};
use dominion_bots::match_runner::{run_match_parallel, Kingdoms};
use dominion_bots::policy::HeuristicBot;
use dominion_bots::Agent;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "models/net.bin".into());
    let pairs: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let net = Net::load(&path).expect("load network");
    let cfg = MctsConfig::default();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!(
        "{path} at {}x{} ({} leaves)\n",
        cfg.worlds,
        cfg.iterations,
        if cfg.use_value_head {
            "value head"
        } else {
            "rollout"
        }
    );

    let net_ref = &net;
    let show = |title: &str, res: dominion_bots::match_runner::MatchResult| {
        let sigma = (res.win_rate_a() - 0.5).abs() / res.stderr().max(1e-9);
        println!("{title}\n  {res}\n  {sigma:.1} standard errors from even\n");
    };

    show(
        "vs the heuristic it uses as prior and rollout:",
        run_match_parallel(
            move || Box::new(NetMctsAgent::new(cfg, net_ref)) as Box<dyn Agent>,
            || Box::new(HeuristicBot) as Box<dyn Agent>,
            pairs,
            0xBEEF,
            &Kingdoms::Random,
            cores,
        ),
    );

    let menu = double_witch();
    show(
        "vs the strongest hand-written menu (Double Witch):",
        run_match_parallel(
            move || Box::new(NetMctsAgent::new(cfg, net_ref)) as Box<dyn Agent>,
            move || Box::new(MenuBot::new(double_witch())) as Box<dyn Agent>,
            pairs,
            0xC0FFEE,
            &Kingdoms::RandomWith(required_kingdom(&menu)),
            cores,
        ),
    );

    show(
        "vs the same search with no network (what training bought):",
        run_match_parallel(
            move || Box::new(NetMctsAgent::new(cfg, net_ref)) as Box<dyn Agent>,
            move || Box::new(MctsAgent::new(cfg)) as Box<dyn Agent>,
            pairs,
            0xD00D,
            &Kingdoms::Random,
            cores,
        ),
    );

    println!(
        "No number here is against the app's Hard bot. Nothing in this project\n\
         has ever been measured against it — the target is still unquantified."
    );
}
