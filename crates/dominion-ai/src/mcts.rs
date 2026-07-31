//! Determinized UCT — perfect-information Monte Carlo search.
//!
//! The hidden information in Dominion is only *where* known cards are, never
//! *which* cards exist (see [`dominion_core::determinize`]). So the standard
//! recipe applies well here: sample several concrete worlds consistent with what
//! the player can see, run an ordinary perfect-information UCT search in each,
//! and pick the move the ensemble visited most.
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
    pub seed: u64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            worlds: 8,
            iterations: 400,
            exploration: 2.5,
            skip_trivial: true,
            seed: 0x5EED,
        }
    }
}

const NO_CHILD: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Edge {
    mv: Move,
    child: u32,
    /// Prior probability of this move, from [`prior::priors`].
    p: f32,
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

    fn ensure_edges(&mut self, node: usize, state: &GameState, d: &Decision, eval: &dyn Evaluator) {
        if self.nodes[node].edges.is_empty() {
            self.nodes[node].player = d.player as u8;
            let options = prior::restrict(state, d);
            let priors = eval.priors(state, d, &options);
            self.nodes[node].edges = options
                .iter()
                .zip(priors)
                .map(|(&mv, p)| Edge {
                    mv,
                    child: NO_CHILD,
                    p,
                })
                .collect();
        }
    }

    /// PUCT: `Q + c * P * sqrt(N_parent) / (1 + N_child)`.
    ///
    /// Unvisited edges are not forced to the front the way plain UCT does it —
    /// with a branching factor of 15 and a few hundred iterations, expanding
    /// every child once would spend the entire budget before learning
    /// anything. The prior decides what is worth a first look.
    fn select(&self, node: usize, exploration: f32) -> usize {
        let n = &self.nodes[node];
        let me = n.player as usize;
        let sqrt_parent = (n.visits.max(1) as f32).sqrt();
        // First-play urgency: judge an unseen move by how the parent is doing,
        // rather than by an optimistic zero.
        let parent_q = n.sum[me] / n.visits.max(1) as f32;

        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (i, e) in n.edges.iter().enumerate() {
            let (q, child_visits) = if e.child == NO_CHILD {
                (parent_q, 0u32)
            } else {
                let c = &self.nodes[e.child as usize];
                (c.sum[me] / c.visits.max(1) as f32, c.visits)
            };
            let u = exploration * e.p * sqrt_parent / (1.0 + child_visits as f32);
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

/// One PUCT iteration in a fixed (fully known) world.
fn iterate(tree: &mut Tree, root: &GameState, exploration: f32, eval: &dyn Evaluator) {
    let mut game = Game {
        state: root.clone(),
    };
    let mut path: Vec<usize> = vec![0];
    let mut node = 0usize;

    loop {
        if game.is_over() {
            break;
        }
        let d = game.decision().expect("live game has a decision").clone();
        tree.ensure_edges(node, &game.state, &d, eval);

        let edge_idx = tree.select(node, exploration);
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

            for &n in &path {
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
    for &n in &path {
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

/// Search the position and return the best move, plus the ensemble's visit
/// counts for inspection.
pub fn search(
    state: &GameState,
    d: &Decision,
    cfg: &MctsConfig,
    rng: &mut Rng,
) -> (Move, Vec<(Move, u32)>) {
    search_with(state, d, cfg, &HeuristicEvaluator, rng)
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
    // Apply the same restriction the tree uses, so the root agrees with its
    // own children about what is worth considering.
    let options = prior::restrict(state, d);
    if options.len() == 1 {
        return (options[0], vec![(options[0], 1)]);
    }
    let mut totals: Vec<(Move, u32)> = options.iter().map(|&m| (m, 0)).collect();

    for _ in 0..cfg.worlds {
        let world = determinize(state, d.player, rng);
        let mut tree = Tree::new(d.player);
        for _ in 0..cfg.iterations {
            iterate(&mut tree, &world, cfg.exploration, eval);
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
    }

    let best = totals
        .iter()
        .max_by_key(|(_, v)| *v)
        .map(|(m, _)| *m)
        .unwrap_or(options[0]);
    (best, totals)
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
        let label = format!("NetMCTS({}x{})", cfg.worlds, cfg.iterations);
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
        let eval = crate::evaluator::NetEvaluator { net: self.net };
        search_with(state, d, &self.cfg, &eval, &mut self.rng).0
    }

    fn name(&self) -> String {
        self.label.clone()
    }
}
