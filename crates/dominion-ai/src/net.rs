//! A small, dependency-free MLP: the policy/value network that replaces the
//! heuristic as MCTS's prior and, eventually, its leaf evaluation.
//!
//! No autodiff crate, no BLAS — the network is a few hundred neurons wide, so
//! hand-written forward and backward passes over `Vec<f32>` are simpler to
//! reason about than pulling in a dependency for it, and keep the whole
//! project buildable offline with only `rustc`'s own toolchain.
//!
//! Architecture: a shared trunk (two ReLU layers) feeding two heads — a policy
//! head producing logits over the fixed [`dominion_core::MOVE_SPACE`], and a
//! value head producing a win-probability estimate for the player to move.
//! Both heads are trained together (policy against MCTS visit counts, value
//! against the eventual game outcome), which is the standard AlphaZero-style
//! split: the policy head is what `prior::priors` becomes, the value head is
//! what lets a search skip playing a rollout out to the end.

use dominion_core::state::MOVE_SPACE;
use dominion_core::Rng;

use crate::features::FEATURE_DIM;

/// Default hidden widths. These are only defaults: a [`Net`] carries its own
/// widths, and a weights file records them, so networks of different sizes
/// coexist and older checkpoints keep loading after the default changes.
pub const DEFAULT_HIDDEN1: usize = 128;
pub const DEFAULT_HIDDEN2: usize = 64;

fn rand_f32(rng: &mut Rng, scale: f32) -> f32 {
    // next_u64's top 24 bits give plenty of mantissa for an f32 in [-scale, scale).
    let u = (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
    (u * 2.0 - 1.0) * scale
}

/// A single fully-connected layer, `y = W x + b`, stored row-major
/// (`w[o * in_dim + i]`).
#[derive(Clone, Debug)]
struct Layer {
    w: Vec<f32>,
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl Layer {
    fn new(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Self {
        // He initialization: keeps activation variance roughly constant
        // through a ReLU trunk regardless of layer width.
        let scale = (2.0 / in_dim as f32).sqrt();
        Layer {
            w: (0..in_dim * out_dim).map(|_| rand_f32(rng, scale)).collect(),
            b: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        let mut y = self.b.clone();
        for o in 0..self.out_dim {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            y[o] += row.iter().zip(x).map(|(w, x)| w * x).sum::<f32>();
        }
        y
    }

    /// Backprop one example: given `dL/dy`, update this layer's weights in
    /// place with plain online SGD and return `dL/dx` for the caller to keep
    /// propagating backward.
    fn backward(&mut self, x: &[f32], grad_y: &[f32], lr: f32) -> Vec<f32> {
        let mut grad_x = vec![0.0f32; self.in_dim];
        for o in 0..self.out_dim {
            let g = grad_y[o];
            if g == 0.0 {
                continue;
            }
            let row = &mut self.w[o * self.in_dim..(o + 1) * self.in_dim];
            for i in 0..self.in_dim {
                grad_x[i] += row[i] * g;
                row[i] -= lr * g * x[i];
            }
            self.b[o] -= lr * g;
        }
        grad_x
    }

    fn write(&self, out: &mut Vec<u8>) {
        for &v in &self.w {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.b {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn read(in_dim: usize, out_dim: usize, bytes: &[u8], pos: &mut usize) -> Option<Self> {
        let read_f32 = |bytes: &[u8], pos: &mut usize| -> Option<f32> {
            let chunk = bytes.get(*pos..*pos + 4)?;
            *pos += 4;
            Some(f32::from_le_bytes(chunk.try_into().ok()?))
        };
        let w = (0..in_dim * out_dim)
            .map(|_| read_f32(bytes, pos))
            .collect::<Option<Vec<f32>>>()?;
        let b = (0..out_dim)
            .map(|_| read_f32(bytes, pos))
            .collect::<Option<Vec<f32>>>()?;
        Some(Layer {
            w,
            b,
            in_dim,
            out_dim,
        })
    }
}

#[derive(Clone, Copy)]
enum LayerId {
    Trunk1,
    Trunk2,
    Policy,
    Value,
}

fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

fn relu_backward(pre: &[f32], grad_y: &[f32]) -> Vec<f32> {
    pre.iter()
        .zip(grad_y)
        .map(|(&p, &g)| if p > 0.0 { g } else { 0.0 })
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A forward pass, retaining what the backward pass needs.
struct Forward {
    x: Vec<f32>,
    h1_pre: Vec<f32>,
    h1: Vec<f32>,
    h2_pre: Vec<f32>,
    h2: Vec<f32>,
    policy_logits: [f32; MOVE_SPACE],
    value: f32,
}

/// The policy/value network.
#[derive(Clone)]
pub struct Net {
    trunk1: Layer,
    trunk2: Layer,
    policy_head: Layer,
    value_head: Layer,
}

impl Net {
    /// A network at the default width.
    pub fn new(rng: &mut Rng) -> Self {
        Net::with_hidden(DEFAULT_HIDDEN1, DEFAULT_HIDDEN2, rng)
    }

    /// A network at an explicit width, for capacity experiments.
    pub fn with_hidden(h1: usize, h2: usize, rng: &mut Rng) -> Self {
        Net {
            trunk1: Layer::new(FEATURE_DIM, h1, rng),
            trunk2: Layer::new(h1, h2, rng),
            policy_head: Layer::new(h2, MOVE_SPACE, rng),
            value_head: Layer::new(h2, 1, rng),
        }
    }

    /// `(hidden1, hidden2)` for this network.
    pub fn hidden(&self) -> (usize, usize) {
        (self.trunk1.out_dim, self.trunk2.out_dim)
    }

    /// Total learnable parameters, the honest measure of capacity.
    pub fn parameters(&self) -> usize {
        [&self.trunk1, &self.trunk2, &self.policy_head, &self.value_head]
            .iter()
            .map(|l| l.w.len() + l.b.len())
            .sum()
    }

    fn forward(&self, x: &[f32; FEATURE_DIM]) -> Forward {
        let h1_pre = self.trunk1.forward(x);
        let h1 = relu(&h1_pre);
        let h2_pre = self.trunk2.forward(&h1);
        let h2 = relu(&h2_pre);
        let logits_vec = self.policy_head.forward(&h2);
        let mut policy_logits = [0.0f32; MOVE_SPACE];
        policy_logits.copy_from_slice(&logits_vec);
        let value = sigmoid(self.value_head.forward(&h2)[0]);
        Forward {
            x: x.to_vec(),
            h1_pre,
            h1,
            h2_pre,
            h2,
            policy_logits,
            value,
        }
    }

    /// Softmax over `legal_indices` only, ignoring every other logit — this is
    /// what makes the policy head answer "how should probability be split
    /// among *these* options", matching how the search actually uses it.
    pub fn policy_over(&self, x: &[f32; FEATURE_DIM], legal_indices: &[usize]) -> Vec<f32> {
        let f = self.forward(x);
        softmax_subset(&f.policy_logits, legal_indices)
    }

    pub fn value(&self, x: &[f32; FEATURE_DIM]) -> f32 {
        self.forward(x).value
    }

    /// One online SGD step. `legal_indices`/`target_probs` describe the
    /// masked softmax target for the policy head (same length, same order);
    /// `value_target` is the eventual game result for the deciding player, in
    /// `[0, 1]`. Returns `(policy_loss, value_loss)` for monitoring.
    pub fn train_step(
        &mut self,
        x: &[f32; FEATURE_DIM],
        legal_indices: &[usize],
        target_probs: &[f32],
        value_target: f32,
        lr: f32,
    ) -> (f32, f32) {
        let f = self.forward(x);

        // --- policy head: masked softmax cross-entropy -------------------
        let probs = softmax_subset(&f.policy_logits, legal_indices);
        let policy_loss = -legal_indices
            .iter()
            .zip(target_probs)
            .zip(&probs)
            .map(|((_, &t), &p)| if t > 0.0 { t * p.max(1e-9).ln() } else { 0.0 })
            .sum::<f32>();

        // dL/dz_i = p_i - t_i for i in the masked subset, 0 elsewhere: logits
        // outside the mask did not take part in this example's softmax.
        let mut grad_logits = vec![0.0f32; MOVE_SPACE];
        for ((&idx, &t), &p) in legal_indices.iter().zip(target_probs).zip(&probs) {
            grad_logits[idx] = p - t;
        }

        // --- value head: MSE on a sigmoid output --------------------------
        let value_loss = (f.value - value_target).powi(2);
        let grad_value_pre = 2.0 * (f.value - value_target) * f.value * (1.0 - f.value);

        // --- backward, accumulating both heads' gradient into the trunk ---
        let grad_h2_from_policy = self.policy_head.backward(&f.h2, &grad_logits, lr);
        let grad_h2_from_value = self.value_head.backward(&f.h2, &[grad_value_pre], lr);
        let grad_h2: Vec<f32> = grad_h2_from_policy
            .iter()
            .zip(&grad_h2_from_value)
            .map(|(a, b)| a + b)
            .collect();

        let grad_h2_pre = relu_backward(&f.h2_pre, &grad_h2);
        let grad_h1 = self.trunk2.backward(&f.h1, &grad_h2_pre, lr);
        let grad_h1_pre = relu_backward(&f.h1_pre, &grad_h1);
        let _ = self.trunk1.backward(&f.x, &grad_h1_pre, lr);

        (policy_loss, value_loss)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let (h1, h2) = self.hidden();
        let mut out = Vec::new();
        for dim in [
            FEATURE_DIM as u32,
            h1 as u32,
            h2 as u32,
            MOVE_SPACE as u32,
        ] {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        self.trunk1.write(&mut out);
        self.trunk2.write(&mut out);
        self.policy_head.write(&mut out);
        self.value_head.write(&mut out);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let read_u32 = |bytes: &[u8], pos: &mut usize| -> Option<u32> {
            let chunk = bytes.get(*pos..*pos + 4)?;
            *pos += 4;
            Some(u32::from_le_bytes(chunk.try_into().ok()?))
        };
        let mut pos = 0;
        let feature_dim = read_u32(bytes, &mut pos)? as usize;
        let h1 = read_u32(bytes, &mut pos)? as usize;
        let h2 = read_u32(bytes, &mut pos)? as usize;
        let move_space = read_u32(bytes, &mut pos)? as usize;

        // The hidden widths are whatever the file says, so checkpoints of
        // different sizes all load. The input and output widths are fixed by
        // the game encoding, so a mismatch there really is a foreign file.
        if feature_dim != FEATURE_DIM || move_space != MOVE_SPACE {
            return None;
        }
        // Guard against a corrupt header driving an enormous allocation.
        if h1 == 0 || h2 == 0 || h1 > 65536 || h2 > 65536 {
            return None;
        }
        Some(Net {
            trunk1: Layer::read(FEATURE_DIM, h1, bytes, &mut pos)?,
            trunk2: Layer::read(h1, h2, bytes, &mut pos)?,
            policy_head: Layer::read(h2, MOVE_SPACE, bytes, &mut pos)?,
            value_head: Layer::read(h2, 1, bytes, &mut pos)?,
        })
    }

    /// Average several networks into one, weighted by how much data each was
    /// trained on.
    ///
    /// This is federated averaging, and it is only meaningful under one
    /// condition: every input must have been fine-tuned from the *same*
    /// starting checkpoint. Then each is that checkpoint plus a gradient
    /// update computed on its own slice of data, and averaging them
    /// approximates one large-batch update over all the slices — which is the
    /// point of letting several machines generate self-play at once.
    ///
    /// Inputs must also share a width; mixing sizes returns `None`.
    ///
    /// Averaging networks trained from *different* random initialisations is a
    /// different and much worse proposition: hidden units are only meaningful
    /// relative to their own network, so unit 7 of one has nothing to do with
    /// unit 7 of another and averaging them destroys both. The caller is
    /// responsible for that precondition; nothing here can check it.
    pub fn weighted_average(nets: &[(Net, f32)]) -> Option<Net> {
        let (first, _) = nets.first()?;
        let total: f32 = nets.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            return None;
        }

        if nets.iter().any(|(n, _)| n.hidden() != first.hidden()) {
            return None;
        }
        let mut out = first.clone();
        for layer in [
            LayerId::Trunk1,
            LayerId::Trunk2,
            LayerId::Policy,
            LayerId::Value,
        ] {
            let (w_len, b_len) = {
                let l = out.layer(layer);
                (l.w.len(), l.b.len())
            };
            let mut w_acc = vec![0.0f32; w_len];
            let mut b_acc = vec![0.0f32; b_len];
            for (net, weight) in nets {
                let l = net.layer(layer);
                if l.w.len() != w_len || l.b.len() != b_len {
                    return None; // mismatched architecture
                }
                let share = weight / total;
                for (acc, v) in w_acc.iter_mut().zip(&l.w) {
                    *acc += share * v;
                }
                for (acc, v) in b_acc.iter_mut().zip(&l.b) {
                    *acc += share * v;
                }
            }
            let l = out.layer_mut(layer);
            l.w = w_acc;
            l.b = b_acc;
        }
        Some(out)
    }

    fn layer(&self, id: LayerId) -> &Layer {
        match id {
            LayerId::Trunk1 => &self.trunk1,
            LayerId::Trunk2 => &self.trunk2,
            LayerId::Policy => &self.policy_head,
            LayerId::Value => &self.value_head,
        }
    }

    fn layer_mut(&mut self, id: LayerId) -> &mut Layer {
        match id {
            LayerId::Trunk1 => &mut self.trunk1,
            LayerId::Trunk2 => &mut self.trunk2,
            LayerId::Policy => &mut self.policy_head,
            LayerId::Value => &mut self.value_head,
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
            .ok_or_else(|| std::io::Error::other("weights file does not match this architecture"))
    }
}

fn softmax_subset(logits: &[f32; MOVE_SPACE], indices: &[usize]) -> Vec<f32> {
    let max = indices
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = indices.iter().map(|&i| (logits[i] - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|e| e / sum.max(1e-9)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zeros() -> [f32; FEATURE_DIM] {
        [0.0; FEATURE_DIM]
    }

    #[test]
    fn forward_shapes_are_correct() {
        let mut rng = Rng::new(1);
        let net = Net::new(&mut rng);
        let x = zeros();
        let f = net.forward(&x);
        assert_eq!(f.policy_logits.len(), MOVE_SPACE);
        assert!((0.0..=1.0).contains(&f.value));
    }

    #[test]
    fn masked_softmax_sums_to_one_over_the_subset() {
        let mut rng = Rng::new(2);
        let net = Net::new(&mut rng);
        let x = zeros();
        let legal = [3usize, 10, 55, 90];
        let probs = net.policy_over(&x, &legal);
        assert_eq!(probs.len(), legal.len());
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum was {sum}");
        assert!(probs.iter().all(|&p| p >= 0.0));
    }

    /// Training on one fixed example many times should drive the value
    /// prediction toward the target and concentrate policy mass on the target
    /// move — the basic sanity check that backprop is wired correctly.
    #[test]
    fn training_reduces_loss_on_a_single_example() {
        let mut rng = Rng::new(3);
        let mut net = Net::new(&mut rng);
        let mut x = zeros();
        x[0] = 0.7;
        x[5] = -0.3;
        let legal = [1usize, 2, 3];
        let target_probs = vec![0.0, 1.0, 0.0]; // all mass on index 2
        let value_target = 1.0;

        let (p0, v0) = net.train_step(&x, &legal, &target_probs, value_target, 0.05);
        for _ in 0..400 {
            net.train_step(&x, &legal, &target_probs, value_target, 0.05);
        }
        let (p1, v1) = net.train_step(&x, &legal, &target_probs, value_target, 0.05);

        assert!(p1 < p0, "policy loss should drop: {p0} -> {p1}");
        assert!(v1 < v0, "value loss should drop: {v0} -> {v1}");
        assert!(v1 < 0.05, "value should be near the target, got loss {v1}");

        let probs = net.policy_over(&x, &legal);
        assert!(
            probs[1] > 0.9,
            "policy should concentrate on the trained target, got {probs:?}"
        );
    }

    #[test]
    fn weights_survive_a_byte_round_trip() {
        let mut rng = Rng::new(4);
        let mut net = Net::new(&mut rng);
        // Nudge it away from its initial state so the test cannot pass by
        // accident on an all-zero or freshly-initialized network.
        let x = zeros();
        net.train_step(&x, &[0, 1], &[0.5, 0.5], 0.5, 0.01);

        let bytes = net.to_bytes();
        let restored = Net::from_bytes(&bytes).expect("round trip parses");

        let probe = {
            let mut x = zeros();
            x[10] = 1.0;
            x
        };
        assert_eq!(net.value(&probe), restored.value(&probe));
        assert_eq!(
            net.policy_over(&probe, &[0, 5, 20]),
            restored.policy_over(&probe, &[0, 5, 20])
        );
    }

    #[test]
    fn a_foreign_byte_blob_is_rejected_not_panicked_on() {
        assert!(Net::from_bytes(&[1, 2, 3]).is_none());
    }

    /// Widths live in the file, so a checkpoint saved at one size still loads
    /// after the default changes. Without this, raising the default would
    /// silently strand every network already trained and pushed.
    #[test]
    fn checkpoints_of_any_width_round_trip() {
        for (h1, h2) in [(128usize, 64usize), (512, 256), (32, 16)] {
            let mut rng = Rng::new(h1 as u64);
            let net = Net::with_hidden(h1, h2, &mut rng);
            assert_eq!(net.hidden(), (h1, h2));

            let back = Net::from_bytes(&net.to_bytes()).expect("round trips");
            assert_eq!(back.hidden(), (h1, h2));

            let mut probe = zeros();
            probe[7] = 0.5;
            assert_eq!(net.value(&probe), back.value(&probe));
        }
    }

    #[test]
    fn a_wider_network_really_has_more_capacity() {
        let mut rng = Rng::new(1);
        let small = Net::with_hidden(128, 64, &mut rng);
        let big = Net::with_hidden(512, 256, &mut rng);
        // 32,741 against 228,965 - about 7x. Most of the growth is the h1 x h2
        // matrix, so widening the second layer buys capacity faster than
        // widening the first.
        assert!(
            big.parameters() > 5 * small.parameters(),
            "{} vs {}",
            big.parameters(),
            small.parameters()
        );
        assert_eq!(small.parameters(), 32_741);
        assert_eq!(big.parameters(), 228_965);
    }

    #[test]
    fn merging_different_widths_is_refused() {
        let mut rng = Rng::new(2);
        let a = Net::with_hidden(128, 64, &mut rng);
        let b = Net::with_hidden(512, 256, &mut rng);
        assert!(Net::weighted_average(&[(a, 1.0), (b, 1.0)]).is_none());
    }

    /// Averaging is elementwise on every parameter, so a probe input should
    /// land between what the inputs predict, and equal weights should give the
    /// exact midpoint.
    #[test]
    fn averaging_blends_the_inputs() {
        let mut rng = Rng::new(10);
        let base = Net::new(&mut rng);

        // Two divergent fine-tunes of one shared checkpoint — the situation
        // weighted_average is actually for.
        let mut a = base.clone();
        let mut b = base.clone();
        let x = zeros();
        for _ in 0..200 {
            a.train_step(&x, &[0, 1], &[1.0, 0.0], 1.0, 0.05);
            b.train_step(&x, &[0, 1], &[0.0, 1.0], 0.0, 0.05);
        }

        let va = a.value(&x);
        let vb = b.value(&x);
        assert!(va > vb, "the two fine-tunes should disagree: {va} vs {vb}");

        let avg = Net::weighted_average(&[(a.clone(), 1.0), (b.clone(), 1.0)]).unwrap();
        let vavg = avg.value(&x);
        assert!(
            vavg > vb && vavg < va,
            "average {vavg} should sit between {vb} and {va}"
        );

        // Weighting all the way to one side reproduces that side exactly.
        let just_a = Net::weighted_average(&[(a.clone(), 1.0), (b.clone(), 0.0)]).unwrap();
        assert!((just_a.value(&x) - va).abs() < 1e-5);
    }

    #[test]
    fn averaging_rejects_degenerate_input() {
        let mut rng = Rng::new(11);
        let n = Net::new(&mut rng);
        assert!(Net::weighted_average(&[]).is_none());
        assert!(Net::weighted_average(&[(n, 0.0)]).is_none());
    }
}
