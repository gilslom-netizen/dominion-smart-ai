//! Information Set MCTS (ISMCTS) with a PUCT selection rule.
//!
//! The hidden information in Dominion is only *where* known cards are, never
//! *which* cards exist (see [`dominion_core::determinize`]). So a
//! determinization-based search is a good fit — but *how* the determinizations
//! are combined matters a lot.
//!
//! This started as plain PIMC: N independent worlds, an independent tree in
//! each, visit counts summed at the end. That wastes most of the budget. With 8
//! worlds x 400 iterations, every node was backed by at most 400 samples even
//! though 3200 were paid for, and the measured visit counts were correspondingly
//! noisy.
//!
//! ISMCTS instead grows **one** tree and re-determinizes on every iteration, so
//! all 3200 iterations accumulate into the same statistics. A tree node is an
//! information set rather than a state: it holds every move seen under any
//! determinization, and each iteration may only descend through the subset that
//! its own world makes legal.
//!
//! Two details this needs, both standard:
//!
//! * **Availability counts.** A move that is only legal in a tenth of the worlds
//!   gets a tenth of the chances to be tried, so comparing it against a
//!   universally-legal sibling on raw visits would understate it. Exploration is
//!   therefore scaled by how often each move was actually available, not by the
//!   parent's total visits.
//! * **Lazy edge discovery.** Nodes gain edges as new determinizations reveal
//!   new legal moves, rather than being fixed at first visit.
//!
//! Dominion is unusually friendly to this: the deciding player's own hand is
//! known to them, so at the root every determinization offers exactly the same
//! moves. Move sets only start to diverge deeper in the tree, once cards have
//! been drawn.
//!
//! Rollouts use the heuristic policy rather than random play. That is not an
//! optimisation, it is a requirement: a uniformly random buy phase never builds
//! an economy, so random rollouts in Dominion score every line as roughly
//! equally hopeless and carry almost no signal.

use dominion_bots::policy;
use dominion_bots::Agent;
use dominion_core::state::MAX_PLAYERS;
use dominion_core::{determinize, Decision, Game, GameState, Move, Rng};

use crate::evaluator::{Evaluator, HeuristicEvaluator};
use crate::prior;

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    /// How many hidden-information worlds to sample.
    pub worlds: u32,
    /// UCT iterations spent inside each world.
    pub iterations: u32,
    /// PUCT exploration constant. Results are in [0,1] and the prior is
    /// already concentrated, so this scales how willing the search is to
    /// disagree with the prior.
    pub exploration: f32,
    /// Give up searching decisions that cannot matter much, to spend the
    /// budget where it counts.
    pub skip_trivial: bool,
    /// Flattening applied to a *network* prior before the search uses it; see
    /// [`crate::evaluator::NetEvaluator::temperature`]. 1.0 leaves it as
    /// trained. Ignored by the heuristic-prior agent, which has its own
    /// concentration built into `prior::priors`.
    pub prior_temperature: f32,
    /// Price leaves with the network's value head (`true`) or with a
    /// heuristic rollout (`false`).
    ///
    /// Defaults to the rollout, which measured both better calibrated (Brier
    /// 0.1301 against 0.1599) and stronger: 64.17% ± 4.38 at equal search,
    /// and 57.00% ± 2.21 over 500 games when the value head is handed 4.3x
    /// the simulations to spend the same wall clock. It wins on accuracy per
    /// unit of time, not just per simulation, which is what makes it the
    /// right default rather than merely the more accurate option.
    ///
    /// The value head stays selectable because it is ~4.3x cheaper per leaf,
    /// which still matters wherever throughput beats per-decision quality.
    pub use_value_head: bool,
    pub seed: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            worlds: 8,
            iterations: 400,
            exploration: 2.5,
            skip_trivial: true,
            prior_temperature: 1.0,
            use_value_head: false,
            seed: 0x5EED,
        }
    }
}

const NO_CHILD: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Edge {
    mv: Move,
    child: u32,
    /// Prior probability of this move, from the evaluator.
    p: f32,
    /// How many iterations reached this node in a world where this move was
    /// legal. ISMCTS compares moves by availability, not by parent visits.
    avail: u32,
}

struct Node {
    /// Whose decision this node represents.
    player: u8,
    edges: Vec<Edge>,
    visits: u32,
    /// Sum of rollout results, per player.
    sum: [f32; MAX_PLAYERS],
}

impl Node {
    fn new(player: usize) -> Self {
        Node {
            player: player as u8,
            edges: Vec::new(),
            visits: 0,
            sum: [0.0; MAX_PLAYERS],
        }
    }
}

struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn new(root_player: usize) -> Self {
        Tree {
            nodes: vec![Node::new(root_player)],
        }
    }

    /// Bring `node`'s edges up to date with what this determinization allows,
    /// and report which of them are legal right now.
    ///
    /// Newly-seen moves are appended rather than replacing what is there: the
    /// node is an information set, so it accumulates the union of every move
    /// any world has offered, while a single iteration only ever descends
    /// through its own world's subset.
    fn sync_edges(
        &mut self,
        node: usize,
        state: &GameState,
        d: &Decision,
        eval: &dyn Evaluator,
        legal_out: &mut Vec<usize>,
    ) {
        let options = prior::restrict(state, d);
        self.nodes[node].player = d.player as u8;

        let unseen = options
            .iter()
            .any(|mv| !self.nodes[node].edges.iter().any(|e| e.mv == *mv));
        if unseen {
            // Only pay for the evaluator when there is something new to price.
            let priors = eval.priors(state, d, &options);
            for (&mv, &p) in options.iter().zip(&priors) {
                match self.nodes[node].edges.iter_mut().find(|e| e.mv == mv) {
                    Some(e) => e.p = p,
                    None => self.nodes[node].edges.push(Edge {
                        mv,
                        child: NO_CHILD,
                        p,
                        avail: 0,
                    }),
                }
            }
        }

        legal_out.clear();
        for (i, e) in self.nodes[node].edges.iter_mut().enumerate() {
            if options.contains(&e.mv) {
                e.avail += 1;
                legal_out.push(i);
            }
        }
    }

    /// PUCT over the moves this determinization makes legal:
    /// `Q + c * P * sqrt(availability) / (1 + N_child)`.
    ///
    /// Unvisited edges are not forced to the front the way plain UCT does it —
    /// with a branching factor of 15 and a few hundred iterations, expanding
    /// every child once would spend the entire budget before learning
    /// anything. The prior decides what is worth a first look.
    fn select(&self, node: usize, legal: &[usize], exploration: f32) -> usize {
        let n = &self.nodes[node];
        let me = n.player as usize;
        // First-play urgency: judge an unseen move by how the parent is doing,
        // rather than by an optimistic zero.
        let parent_q = n.sum[me] / n.visits.max(1) as f32;

        let mut best = legal[0];
        let mut best_score = f32::NEG_INFINITY;
        for &i in legal {
            let e = &n.edges[i];
            let (q, child_visits) = if e.child == NO_CHILD {
                (parent_q, 0u32)
            } else {
                let c = &self.nodes[e.child as usize];
                (c.sum[me] / c.visits.max(1) as f32, c.visits)
            };
            let u =
                exploration * e.p * (e.avail.max(1) as f32).sqrt() / (1.0 + child_visits as f32);
            let score = q + u;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }
}

/// Play the position out with the heuristic policy and return each player's
/// result.
fn rollout(mut game: Game) -> [f32; MAX_PLAYERS] {
    let mut guard = 0u32;
    while !game.is_over() {
        let d = game.decision().expect("live game has a decision").clone();
        let mv = policy::default_move(&game.state, &d);
        // The heuristic is total, but never trust it blindly inside search.
        let mv = if d.options.contains(&mv) {
            mv
        } else {
            d.options[0]
        };
        game.apply(mv).expect("policy move is legal");
        guard += 1;
        if guard > 100_000 {
            break;
        }
    }
    let mut out = [0.0; MAX_PLAYERS];
    for (i, r) in game.state.results().into_iter().enumerate() {
        out[i] = r;
    }
    out
}

/// One ISMCTS iteration: descend the shared tree inside one sampled world.
///
/// `scratch` and `path` are caller-owned so the hot loop does not allocate a
/// fresh Vec per iteration — at a few thousand iterations per decision and tens
/// of searched decisions per game, that adds up.
fn iterate(
    tree: &mut Tree,
    root: &GameState,
    exploration: f32,
    eval: &dyn Evaluator,
    path: &mut Vec<usize>,
    scratch: &mut Vec<usize>,
) {
    let mut game = Game {
        state: root.clone(),
    };
    path.clear();
    path.push(0);
    let mut node = 0usize;

    loop {
        if game.is_over() {
            break;
        }
        let d = game.decision().expect("live game has a decision").clone();
        tree.sync_edges(node, &game.state, &d, eval, scratch);
        if scratch.is_empty() {
            break;
        }

        let edge_idx = tree.select(node, scratch, exploration);
        let edge = tree.nodes[node].edges[edge_idx];
        game.apply(edge.mv).expect("tree move is legal");

        if edge.child == NO_CHILD {
            // Expand one new node. If the evaluator can price this leaf
            // directly, use that instead of playing the game out — that is
            // the entire benefit of a trained value head over a heuristic
            // rollout: an O(1) estimate instead of simulating the rest of
            // the game.
            let next_player = game.decision().map(|d| d.player).unwrap_or(0);
            tree.nodes.push(Node::new(next_player));
            let new_idx = (tree.nodes.len() - 1) as u32;
            tree.nodes[node].edges[edge_idx].child = new_idx;
            path.push(new_idx as usize);

            let result = if game.is_over() {
                let mut out = [0.0; MAX_PLAYERS];
                for (i, r) in game.state.results().into_iter().enumerate() {
                    out[i] = r;
                }
                out
            } else if let Some(v) = eval.leaf_value(&game.state, next_player) {
                let mut out = [0.0; MAX_PLAYERS];
                out[next_player] = v;
                for (i, o) in out.iter_mut().enumerate() {
                    if i != next_player {
                        *o = 1.0 - v;
                    }
                }
                out
            } else {
                rollout(game)
            };

            for &n in path.iter() {
                let node = &mut tree.nodes[n];
                node.visits += 1;
                for p in 0..MAX_PLAYERS {
                    node.sum[p] += result[p];
                }
            }
            return;
        }
        node = edge.child as usize;
        path.push(node);
    }

    // The in-tree walk itself reached a finished game (can happen once the
    // tree is deep enough), so the actual result is exact, not a rollout.
    let mut result = [0.0; MAX_PLAYERS];
    for (i, r) in game.state.results().into_iter().enumerate() {
        result[i] = r;
    }
    for &n in path.iter() {
        let node = &mut tree.nodes[n];
        node.visits += 1;
        for p in 0..MAX_PLAYERS {
            node.sum[p] += result[p];
        }
    }
}

/// Decisions where searching cannot plausibly pay for itself: the heuristic
/// answer is either forced or overwhelmingly likely to be right.
fn is_trivial(d: &Decision) -> bool {
    use dominion_core::Ctx::*;
    match d.ctx {
        // Never wrong in the Base set.
        MoatReveal => true,
        // Choosing which of two identical-looking junk cards to dump.
        SentryDiscard | SentryOrder | BureaucratReveal => d.options.len() <= 2,
        _ => false,
    }
}

/// Everything a search produced: the move, the visit distribution behind it,
/// and the search's own opinion of the position.
#[derive(Clone, Debug)]
pub struct SearchOutcome {
    pub best: Move,
    /// Visits per move, in the order they were offered.
    pub visits: Vec<(Move, u32)>,
    /// The search's estimate of the deciding player's win probability, in
    /// `[0, 1]`. This is the average result over every simulation that passed
    /// through the root, so it is a far better estimate than a raw network
    /// forward pass — which is exactly what makes it useful as a bootstrap
    /// target when training the value head.
    pub value: f32,
}

/// Search the position and return the best move, plus the visit counts for
/// inspection.
pub fn search(
    state: &GameState,
    d: &Decision,
    cfg: &MctsConfig,
    rng: &mut Rng,
) -> (Move, Vec<(Move, u32)>) {
    let out = search_full(state, d, cfg, &HeuristicEvaluator, rng);
    (out.best, out.visits)
}

/// As [`search`], but steered by an arbitrary [`Evaluator`] — a trained
/// network, a blend of network and heuristic, or the heuristic itself.
pub fn search_with(
    state: &GameState,
    d: &Decision,
    cfg: &MctsConfig,
    eval: &dyn Evaluator,
    rng: &mut Rng,
) -> (Move, Vec<(Move, u32)>) {
    let out = search_full(state, d, cfg, eval, rng);
    (out.best, out.visits)
}

/// The full search, including the root's value estimate.
pub fn search_full(
    state: &GameState,
    d: &Decision,
    cfg: &MctsConfig,
    eval: &dyn Evaluator,
    rng: &mut Rng,
) -> SearchOutcome {
    // Apply the same restriction the tree uses, so the root agrees with its
    // own children about what is worth considering.
    let options = prior::restrict(state, d);
    if options.len() == 1 {
        // Forced move: no search happened, so there is no searched value to
        // report. Fall back to whatever the evaluator thinks, or an even
        // position if it has no opinion.
        let value = eval.leaf_value(state, d.player).unwrap_or(0.5);
        return SearchOutcome {
            best: options[0],
            visits: vec![(options[0], 1)],
            value,
        };
    }
    let mut totals: Vec<(Move, u32)> = options.iter().map(|&m| (m, 0)).collect();

    // One tree, re-determinized every iteration. `worlds * iterations` keeps
    // the same total budget the old per-world loop spent, but every sample now
    // lands in the same statistics instead of being split across N trees.
    let mut tree = Tree::new(d.player);
    let mut path: Vec<usize> = Vec::with_capacity(64);
    let mut scratch: Vec<usize> = Vec::with_capacity(32);
    let budget = (cfg.worlds as u64 * cfg.iterations as u64).max(1);

    for _ in 0..budget {
        let world = determinize(state, d.player, rng);
        iterate(
            &mut tree,
            &world,
            cfg.exploration,
            eval,
            &mut path,
            &mut scratch,
        );
    }

    for e in &tree.nodes[0].edges {
        if e.child == NO_CHILD {
            continue;
        }
        let visits = tree.nodes[e.child as usize].visits;
        if let Some(slot) = totals.iter_mut().find(|(m, _)| *m == e.mv) {
            slot.1 += visits;
        }
    }

    let best = totals
        .iter()
        .max_by_key(|(_, v)| *v)
        .map(|(m, _)| *m)
        .unwrap_or(options[0]);

    let root = &tree.nodes[0];
    let value = if root.visits > 0 {
        root.sum[d.player] / root.visits as f32
    } else {
        eval.leaf_value(state, d.player).unwrap_or(0.5)
    };

    SearchOutcome {
        best,
        visits: totals,
        value,
    }
}

/// A search agent, ready to drop into the match harness.
pub struct MctsAgent {
    pub cfg: MctsConfig,
    rng: Rng,
    label: String,
}

impl MctsAgent {
    pub fn new(cfg: MctsConfig) -> Self {
        let label = format!("MCTS({}x{})", cfg.worlds, cfg.iterations);
        MctsAgent {
            rng: Rng::new(cfg.seed),
            cfg,
            label,
        }
    }

    pub fn named(mut self, name: &str) -> Self {
        self.label = name.into();
        self
    }
}

impl Agent for MctsAgent {
    fn decide(&mut self, state: &GameState, d: &Decision) -> Move {
        if d.options.len() == 1 {
            return d.options[0];
        }
        if self.cfg.skip_trivial && is_trivial(d) {
            return policy::default_move(state, d);
        }
        search(state, d, &self.cfg, &mut self.rng).0
    }

    fn name(&self) -> String {
        self.label.clone()
    }
}

/// Search steered by a trained network instead of the heuristic, for measuring
/// what the network has actually learned.
pub struct NetMctsAgent<'a> {
    pub cfg: MctsConfig,
    rng: Rng,
    net: &'a crate::net::Net,
    label: String,
}

impl<'a> NetMctsAgent<'a> {
    pub fn new(cfg: MctsConfig, net: &'a crate::net::Net) -> Self {
        let label = if cfg.use_value_head {
            format!("NetMCTS({}x{} valuehead)", cfg.worlds, cfg.iterations)
        } else if (cfg.prior_temperature - 1.0).abs() < 1e-6 {
            format!("NetMCTS({}x{})", cfg.worlds, cfg.iterations)
        } else {
            format!(
                "NetMCTS({}x{} t{} c{})",
                cfg.worlds, cfg.iterations, cfg.prior_temperature, cfg.exploration
            )
        };
        NetMctsAgent {
            rng: Rng::new(cfg.seed),
            cfg,
            net,
            label,
        }
    }
}

impl<'a> Agent for NetMctsAgent<'a> {
    fn decide(&mut self, state: &GameState, d: &Decision) -> Move {
        if d.options.len() == 1 {
            return d.options[0];
        }
        if self.cfg.skip_trivial && is_trivial(d) {
            return policy::default_move(state, d);
        }
        if self.cfg.use_value_head {
            let eval = crate::evaluator::NetEvaluator::with_temperature(
                self.net,
                self.cfg.prior_temperature,
            );
            search_with(state, d, &self.cfg, &eval, &mut self.rng).0
        } else {
            let eval = crate::evaluator::RolloutEvaluator { net: self.net };
            search_with(state, d, &self.cfg, &eval, &mut self.rng).0
        }
    }

    fn name(&self) -> String {
        self.label.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::HeuristicEvaluator;
    use dominion_core::Game;

    fn first_real_decision(seed: u64) -> (GameState, Decision) {
        let kingdom = Game::random_kingdom(&mut Rng::new(seed));
        let mut g = Game::new(&kingdom, 2, seed).unwrap();
        // Walk to a buy decision with several affordable options, which is
        // where determinizations actually differ deeper in the tree.
        loop {
            let d = g.decision().expect("live game").clone();
            let buys = d
                .options
                .iter()
                .filter(|m| matches!(m, Move::Buy(_)))
                .count();
            if d.ctx == dominion_core::Ctx::BuyPhase && buys > 3 {
                return (g.state.clone(), d);
            }
            let mv = prior::restrict(&g.state, &d)[0];
            g.apply(mv).unwrap();
        }
    }

    /// The whole point of ISMCTS over independent-tree PIMC: every iteration
    /// lands in one shared tree, so the root accumulates the *full* budget
    /// rather than budget/worlds.
    #[test]
    fn the_root_accumulates_the_entire_budget() {
        let (state, d) = first_real_decision(11);
        let cfg = MctsConfig {
            worlds: 4,
            iterations: 50,
            ..Default::default()
        };
        let mut rng = Rng::new(1);
        let (_, visits) = search_with(&state, &d, &cfg, &HeuristicEvaluator, &mut rng);
        let total: u32 = visits.iter().map(|(_, v)| v).sum();

        // Every iteration that got past the root backs some root child. A
        // handful terminate at the root itself, so allow slack below, but the
        // total must be far above the 50 a single per-world tree would have
        // given under the old scheme.
        let budget = cfg.worlds * cfg.iterations;
        assert!(
            total > budget / 2,
            "root saw {total} visits of a {budget} budget — statistics are not being shared"
        );
        assert!(
            total <= budget,
            "root cannot have more visits than iterations"
        );
    }

    /// A node is an information set: it holds the union of moves seen under any
    /// determinization, and availability is tracked per move.
    #[test]
    fn nodes_accumulate_moves_across_determinizations() {
        let (state, d) = first_real_decision(7);
        let mut rng = Rng::new(2);
        let mut tree = Tree::new(d.player);
        let mut path = Vec::new();
        let mut scratch = Vec::new();

        for _ in 0..300 {
            let world = dominion_core::determinize(&state, d.player, &mut rng);
            iterate(
                &mut tree,
                &world,
                2.5,
                &HeuristicEvaluator,
                &mut path,
                &mut scratch,
            );
        }

        // The deciding player knows their own hand, so at the root every world
        // offers the same moves and all of them should be universally available.
        let root_avail: Vec<u32> = tree.nodes[0].edges.iter().map(|e| e.avail).collect();
        assert!(!root_avail.is_empty());
        assert!(
            root_avail.iter().all(|&a| a == root_avail[0]),
            "root moves should be equally available across worlds, got {root_avail:?}"
        );

        // Deeper nodes are reached under varying draws, so somewhere in the
        // tree availability must differ between siblings — that is the
        // information-set behaviour availability counts exist to handle.
        let has_varying = tree
            .nodes
            .iter()
            .skip(1)
            .any(|n| n.edges.len() > 1 && n.edges.iter().any(|e| e.avail != n.edges[0].avail));
        assert!(
            has_varying,
            "expected some deeper node to see different moves in different worlds"
        );

        // Priors stay a distribution over whatever the node has discovered.
        for n in &tree.nodes {
            for e in &n.edges {
                assert!(e.p >= 0.0 && e.p <= 1.0, "prior out of range: {}", e.p);
            }
        }
    }

    /// Availability can never exceed the number of times the node was reached.
    #[test]
    fn availability_never_exceeds_node_visits() {
        let (state, d) = first_real_decision(3);
        let mut rng = Rng::new(5);
        let mut tree = Tree::new(d.player);
        let mut path = Vec::new();
        let mut scratch = Vec::new();
        for _ in 0..200 {
            let world = dominion_core::determinize(&state, d.player, &mut rng);
            iterate(
                &mut tree,
                &world,
                2.5,
                &HeuristicEvaluator,
                &mut path,
                &mut scratch,
            );
        }
        for (i, n) in tree.nodes.iter().enumerate() {
            for e in &n.edges {
                // A node is visited once per iteration that reaches it, and a
                // move can be available at most once per such iteration. The
                // root is reached before its visit is recorded, so allow one.
                assert!(
                    e.avail <= n.visits + 1,
                    "node {i}: move {} available {} times but node visited {}",
                    e.mv,
                    e.avail,
                    n.visits
                );
            }
        }
    }
}
