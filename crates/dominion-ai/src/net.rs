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

/// How weights are updated.
///
/// Plain SGD at a fixed learning rate was the only option here for a long
/// time, and it is the last untested component of the system: more data, more
/// capacity and deeper search were each measured and each changed nothing, so
/// the optimiser is what remains.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Optimizer {
    /// `w -= lr * g`. No state, no adaptation.
    Sgd,
    /// Per-parameter adaptive step sizes with momentum. Matters most when
    /// gradients differ wildly in scale across parameters, which is exactly
    /// this network's situation: the policy head sees a gradient on the handful
    /// of legal moves and nothing on the other ninety-odd, while the trunk sees
    /// dense updates from both heads at once.
    Adam,
}

/// Optimiser settings plus the global step counter Adam's bias correction
/// needs.
#[derive(Clone, Copy, Debug)]
pub struct OptConfig {
    pub kind: Optimizer,
    pub lr: f32,
    /// Steps taken so far, 1-based. Bias correction is significant for the
    /// first few thousand steps and negligible after.
    pub step: u64,
}

impl OptConfig {
    pub fn sgd(lr: f32) -> Self {
        OptConfig {
            kind: Optimizer::Sgd,
            lr,
            step: 1,
        }
    }
    /// Adam's usual starting point is an order of magnitude below SGD's,
    /// because the adaptive denominator already scales the step.
    pub fn adam(lr: f32) -> Self {
        OptConfig {
            kind: Optimizer::Adam,
            lr,
            step: 1,
        }
    }
}

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

/// A single fully-connected layer, `y = W x + b`, stored row-major
/// (`w[o * in_dim + i]`).
#[derive(Clone, Debug)]
struct Layer {
    w: Vec<f32>,
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
    /// Adam moment estimates. Empty under SGD, allocated on first use, and
    /// deliberately not serialised — these are optimiser state, not the model,
    /// and checkpoints get shared between machines that may train differently.
    m_w: Vec<f32>,
    v_w: Vec<f32>,
    m_b: Vec<f32>,
    v_b: Vec<f32>,
}

impl Layer {
    fn new(in_dim: usize, out_dim: usize, rng: &mut Rng) -> Self {
        // He initialization: keeps activation variance roughly constant
        // through a ReLU trunk regardless of layer width.
        let scale = (2.0 / in_dim as f32).sqrt();
        Layer {
            w: (0..in_dim * out_dim)
                .map(|_| rand_f32(rng, scale))
                .collect(),
            b: vec![0.0; out_dim],
            in_dim,
            out_dim,
            m_w: Vec::new(),
            v_w: Vec::new(),
            m_b: Vec::new(),
            v_b: Vec::new(),
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        let mut y = self.b.clone();
        for o in 0..self.out_dim {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            y[o] += dot(row, x);
        }
        y
    }

    /// Backprop one example: given `dL/dy`, update this layer's weights in
    /// place with plain online SGD and return `dL/dx` for the caller to keep
    /// propagating backward.
    fn backward(&mut self, x: &[f32], grad_y: &[f32], opt: &OptConfig) -> Vec<f32> {
        let mut grad_x = vec![0.0f32; self.in_dim];
        match opt.kind {
            Optimizer::Sgd => {
                for o in 0..self.out_dim {
                    let g = grad_y[o];
                    if g == 0.0 {
                        continue;
                    }
                    let row = &mut self.w[o * self.in_dim..(o + 1) * self.in_dim];
                    // Two independent vectorizable passes rather than one
                    // interleaved scalar loop: accumulate this row's
                    // contribution to dL/dx, then apply the weight update.
                    axpy(&mut grad_x, row, g);
                    axpy(row, x, -opt.lr * g);
                    self.b[o] -= opt.lr * g;
                }
            }
            Optimizer::Adam => {
                if self.m_w.is_empty() {
                    self.m_w = vec![0.0; self.w.len()];
                    self.v_w = vec![0.0; self.w.len()];
                    self.m_b = vec![0.0; self.b.len()];
                    self.v_b = vec![0.0; self.b.len()];
                }
                let t = opt.step.max(1) as i32;
                let c1 = 1.0 - BETA1.powi(t);
                let c2 = 1.0 - BETA2.powi(t);
                for o in 0..self.out_dim {
                    let g = grad_y[o];
                    if g == 0.0 {
                        continue;
                    }
                    let lo = o * self.in_dim;
                    let row = &mut self.w[lo..lo + self.in_dim];
                    axpy(&mut grad_x, row, g);

                    let mw = &mut self.m_w[lo..lo + self.in_dim];
                    let vw = &mut self.v_w[lo..lo + self.in_dim];
                    for i in 0..self.in_dim {
                        let gi = g * x[i];
                        mw[i] = BETA1 * mw[i] + (1.0 - BETA1) * gi;
                        vw[i] = BETA2 * vw[i] + (1.0 - BETA2) * gi * gi;
                        let mhat = mw[i] / c1;
                        let vhat = vw[i] / c2;
                        row[i] -= opt.lr * mhat / (vhat.sqrt() + EPS);
                    }

                    self.m_b[o] = BETA1 * self.m_b[o] + (1.0 - BETA1) * g;
                    self.v_b[o] = BETA2 * self.v_b[o] + (1.0 - BETA2) * g * g;
                    let mhat = self.m_b[o] / c1;
                    let vhat = self.v_b[o] / c2;
                    self.b[o] -= opt.lr * mhat / (vhat.sqrt() + EPS);
                }
            }
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
            m_w: Vec::new(),
            v_w: Vec::new(),
            m_b: Vec::new(),
            v_b: Vec::new(),
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

/// Dot product with eight independent accumulators.
///
/// The obvious `a.iter().zip(b).map(|(x,y)| x*y).sum()` compiles to a scalar
/// loop and nothing more, because f32 addition is not associative and LLVM will
/// not reorder the running sum without being told it may. Summing into several
/// partial accumulators makes the reassociation explicit in the source, so each
/// one is an independent dependency chain and the whole thing vectorizes.
///
/// This changes the summation order and therefore the last bits of the result.
/// That is fine here — these are neural network activations, not accounting —
/// but it does mean training is no longer bit-reproducible against the old
/// scalar version.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];
    let chunks = a.len() / LANES;

    for c in 0..chunks {
        let base = c * LANES;
        // Fixed-size slices let the bounds checks fold away.
        let (aw, bw) = (&a[base..base + LANES], &b[base..base + LANES]);
        for l in 0..LANES {
            acc[l] += aw[l] * bw[l];
        }
    }

    let mut total = 0.0f32;
    for v in acc {
        total += v;
    }
    for i in chunks * LANES..a.len() {
        total += a[i] * b[i];
    }
    total
}

/// `dst += scale * src`, the inner loop of the backward pass.
#[inline]
fn axpy(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d += scale * s;
    }
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
///
/// Deliberately does *not* keep a copy of the input: the caller already owns
/// it, and cloning 139 floats on every one of a few million training steps is
/// pure waste.
struct Forward {
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
        [
            &self.trunk1,
            &self.trunk2,
            &self.policy_head,
            &self.value_head,
        ]
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
    ///
    /// A position with fewer than two legal moves trains the **value head
    /// only**. Its policy target is a point mass on the sole option, so the
    /// policy gradient is identically zero and computing it is pure waste —
    /// but the position still happened and its value target is as real as any
    /// other. Dropping such examples outright cost 61.7% of the value head's
    /// training data and measurably hurt it.
    pub fn train_step(
        &mut self,
        x: &[f32; FEATURE_DIM],
        legal_indices: &[usize],
        target_probs: &[f32],
        value_target: f32,
        lr: f32,
    ) -> (f32, f32) {
        self.train_step_with(
            x,
            legal_indices,
            target_probs,
            value_target,
            &OptConfig::sgd(lr),
        )
    }

    /// As [`Net::train_step`], with an explicit optimiser.
    pub fn train_step_with(
        &mut self,
        x: &[f32; FEATURE_DIM],
        legal_indices: &[usize],
        target_probs: &[f32],
        value_target: f32,
        opt: &OptConfig,
    ) -> (f32, f32) {
        let f = self.forward(x);
        let trains_policy = legal_indices.len() > 1;

        // --- policy head: masked softmax cross-entropy -------------------
        let mut grad_logits = vec![0.0f32; MOVE_SPACE];
        let policy_loss = if trains_policy {
            let probs = softmax_subset(&f.policy_logits, legal_indices);
            // dL/dz_i = p_i - t_i for i in the masked subset, 0 elsewhere:
            // logits outside the mask did not take part in this softmax.
            for ((&idx, &t), &p) in legal_indices.iter().zip(target_probs).zip(&probs) {
                grad_logits[idx] = p - t;
            }
            -legal_indices
                .iter()
                .zip(target_probs)
                .zip(&probs)
                .map(|((_, &t), &p)| if t > 0.0 { t * p.max(1e-9).ln() } else { 0.0 })
                .sum::<f32>()
        } else {
            0.0
        };

        // --- value head: MSE on a sigmoid output --------------------------
        let value_loss = (f.value - value_target).powi(2);
        let grad_value_pre = 2.0 * (f.value - value_target) * f.value * (1.0 - f.value);

        // --- backward, accumulating both heads' gradient into the trunk ---
        // With an all-zero policy gradient this still costs a pass over the
        // policy head, so skip it outright when there is nothing to learn.
        let grad_h2_from_policy = if trains_policy {
            self.policy_head.backward(&f.h2, &grad_logits, opt)
        } else {
            vec![0.0; f.h2.len()]
        };
        let grad_h2_from_value = self.value_head.backward(&f.h2, &[grad_value_pre], opt);
        let grad_h2: Vec<f32> = grad_h2_from_policy
            .iter()
            .zip(&grad_h2_from_value)
            .map(|(a, b)| a + b)
            .collect();

        let grad_h2_pre = relu_backward(&f.h2_pre, &grad_h2);
        let grad_h1 = self.trunk2.backward(&f.h1, &grad_h2_pre, opt);
        let grad_h1_pre = relu_backward(&f.h1_pre, &grad_h1);
        let _ = self.trunk1.backward(x, &grad_h1_pre, opt);

        (policy_loss, value_loss)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let (h1, h2) = self.hidden();
        let mut out = Vec::new();
        for dim in [FEATURE_DIM as u32, h1 as u32, h2 as u32, MOVE_SPACE as u32] {
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

    /// Adam should reach a given target in far fewer steps than SGD at the
    /// same learning rate. That is the whole reason to have it, and it is the
    /// property that would silently fail if the moment updates or the bias
    /// correction were wrong.
    #[test]
    fn adam_converges_faster_than_sgd_at_equal_lr() {
        let fit = |opt_kind: Optimizer, steps: u64| -> f32 {
            let mut rng = Rng::new(7);
            let mut net = Net::new(&mut rng);
            let mut x = zeros();
            x[0] = 0.5;
            x[9] = -0.25;
            let mut cfg = match opt_kind {
                Optimizer::Sgd => OptConfig::sgd(0.01),
                Optimizer::Adam => OptConfig::adam(0.01),
            };
            for t in 1..=steps {
                cfg.step = t;
                net.train_step_with(&x, &[3, 4, 5], &[0.0, 1.0, 0.0], 0.9, &cfg);
            }
            let probs = net.policy_over(&x, &[3, 4, 5]);
            probs[1]
        };

        let sgd = fit(Optimizer::Sgd, 300);
        let adam = fit(Optimizer::Adam, 300);
        assert!(
            adam > sgd,
            "Adam should fit faster at the same lr: adam {adam:.4} vs sgd {sgd:.4}"
        );
        assert!(
            adam > 0.9,
            "Adam should get most of the way there, got {adam:.4}"
        );
    }

    /// Bias correction is what keeps the first steps from being tiny. Without
    /// it the moment estimates start at zero and the early updates are damped
    /// by roughly (1 - beta), which is exactly when a short run is decided.
    #[test]
    fn adam_moves_meaningfully_on_the_very_first_steps() {
        let mut rng = Rng::new(8);
        let mut net = Net::new(&mut rng);
        let mut x = zeros();
        x[1] = 0.8;
        let before = net.value(&x);

        let mut cfg = OptConfig::adam(0.01);
        for t in 1..=10 {
            cfg.step = t;
            net.train_step_with(&x, &[0, 1], &[0.5, 0.5], 1.0, &cfg);
        }
        let after = net.value(&x);
        assert!(
            after - before > 0.005,
            "ten Adam steps barely moved the value head: {before:.5} -> {after:.5}"
        );
    }

    /// Optimiser state is not part of the model. Checkpoints are shared
    /// between machines that may train differently, so moments must not ride
    /// along in the file — and a network that trained under Adam must still
    /// serialize to exactly the same bytes as its weights imply.
    #[test]
    fn adam_state_is_not_serialized() {
        let mut rng = Rng::new(9);
        let mut a = Net::new(&mut rng);
        let x = {
            let mut v = zeros();
            v[4] = 0.3;
            v
        };
        let mut cfg = OptConfig::adam(0.01);
        for t in 1..=50 {
            cfg.step = t;
            a.train_step_with(&x, &[0, 1], &[1.0, 0.0], 1.0, &cfg);
        }

        let bytes = a.to_bytes();
        let b = Net::from_bytes(&bytes).expect("round trip");
        assert_eq!(a.value(&x), b.value(&x));
        assert_eq!(a.policy_over(&x, &[0, 1, 2]), b.policy_over(&x, &[0, 1, 2]));
        // Reloaded, the moments start fresh — the same size as a network that
        // never saw Adam.
        assert_eq!(bytes.len(), Net::new(&mut Rng::new(1)).to_bytes().len());
    }

    #[test]
    fn dot_matches_a_naive_sum() {
        let mut rng = Rng::new(99);
        for len in [0usize, 1, 7, 8, 9, 64, 139, 511] {
            let a: Vec<f32> = (0..len).map(|_| rand_f32(&mut rng, 2.0)).collect();
            let b: Vec<f32> = (0..len).map(|_| rand_f32(&mut rng, 2.0)).collect();
            let naive: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let fast = dot(&a, &b);
            // Reassociated summation, so exact equality is not expected —
            // only that the difference stays at rounding scale.
            let tol = 1e-4 * naive.abs().max(1.0);
            assert!(
                (naive - fast).abs() <= tol,
                "len {len}: naive {naive} vs dot {fast}"
            );
        }
    }

    #[test]
    fn axpy_matches_a_naive_loop() {
        let mut rng = Rng::new(100);
        for len in [0usize, 1, 5, 8, 33] {
            let src: Vec<f32> = (0..len).map(|_| rand_f32(&mut rng, 2.0)).collect();
            let start: Vec<f32> = (0..len).map(|_| rand_f32(&mut rng, 2.0)).collect();
            let scale = 0.37;
            let mut fast = start.clone();
            axpy(&mut fast, &src, scale);
            for i in 0..len {
                assert!((fast[i] - (start[i] + scale * src[i])).abs() < 1e-6);
            }
        }
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
    /// A single-option position must leave the policy head's own parameters
    /// untouched while still teaching the value head.
    ///
    /// Note what is *not* asserted: that the policy output is unchanged. The
    /// trunk is shared, so the value head's updates flow through it and shift
    /// the policy's predictions as a side effect. That is how a shared trunk
    /// is supposed to work — the value head improving the representation is a
    /// benefit, not a leak — so the check is on the policy head's weights,
    /// which are the thing that must not move.
    #[test]
    fn a_forced_position_trains_only_the_value_head() {
        let mut rng = Rng::new(21);
        let mut net = Net::new(&mut rng);
        let mut x = zeros();
        x[2] = 0.4;

        let head_before = net.policy_head.w.clone();
        let bias_before = net.policy_head.b.clone();
        let v_before = net.value(&x);

        for _ in 0..100 {
            let (pl, _) = net.train_step(&x, &[5], &[1.0], 1.0, 0.05);
            assert_eq!(pl, 0.0, "a forced move has no policy loss");
        }

        assert_eq!(net.policy_head.w, head_before, "policy head weights moved");
        assert_eq!(net.policy_head.b, bias_before, "policy head biases moved");
        assert!(
            net.value(&x) > v_before,
            "the value head should still have learned"
        );
    }

    /// And the ordinary case still updates the policy head, so the guard above
    /// cannot pass by accident on a network that never learns a policy at all.
    #[test]
    fn a_real_choice_does_update_the_policy_head() {
        let mut rng = Rng::new(22);
        let mut net = Net::new(&mut rng);
        // A non-zero input matters here: with an all-zero feature vector every
        // activation is zero, so `w -= lr * g * x` leaves the weight matrices
        // untouched and only biases move. Correct arithmetic, but it would
        // make this test pass or fail for the wrong reason.
        let mut x = zeros();
        x[0] = 0.6;
        x[11] = -0.3;

        let head_before = net.policy_head.w.clone();
        for _ in 0..20 {
            net.train_step(&x, &[0, 1], &[1.0, 0.0], 0.5, 0.05);
        }
        assert_ne!(
            net.policy_head.w, head_before,
            "a genuine choice must move the policy head"
        );
    }

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
