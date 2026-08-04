# dominion-smart-ai

A Dominion AI for the Base set, 2nd edition — built to beat the Hard bot in the
official app by a clear margin.

## Scope

Base 2E only: 26 kingdom cards plus the 7 basic cards. That is a deliberate
choice. Implementing all ~500 published Dominion cards is most of the work in a
Dominion project and none of the interesting part; a fixed 33-card pool is
enough to prove out the search and the training loop, and widening it later is
engine work rather than AI work.

## Layout

| Crate | What it is |
|---|---|
| `dominion-core` | Rules engine, determinization, game logs. No dependencies. |
| `dominion-bots` | The shared heuristic policy, buy menus, match harness. |
| `dominion-ai` | Determinized PUCT search and the entry points that use it. |

## Commands

```sh
cargo test                                  # 87 tests
cargo run --release --bin ladder            # heuristic round robin
cargo run --release --bin bench             # machine check + search strength
cargo run --release --bin advise -- --demo  # ask the AI about a position
cargo run --release --bin play              # play a game against it yourself

cargo run --release --bin selfplay -- --games 3000 --tag <name> --net models/net.bin
cargo run --release --bin train -- --net-in models/net.bin --net-out models/net.bin
cargo run --release --example net_vs_net -- a.bin b.bin 200
```

## Engine design

Three properties drive the design, all of them in service of search and
self-play rather than of a user interface.

**Snapshot-anywhere.** Card effects run on an explicit continuation stack
(`GameState::stack`) instead of calling back into the agent. When an effect
needs input it parks a `Decision` and returns. So `GameState` is a plain value
that can be cloned or serialized *at any decision point* — which is exactly what
MCTS needs to expand a node, and what replaying a game log needs in order to
stop partway and ask "what would you play here?".

**A small, flat action space.** Compound choices are decomposed into loops of
single-card picks: Cellar is "choose a card to discard" repeated, not "choose a
subset". That keeps the move space at a fixed 100 (`play`/`buy`/`select` × 33
cards, plus `done`), small enough for a policy head to index directly, instead
of the combinatorial blowup a subset-choice encoding would give.

**Forced moves auto-resolve.** A decision with one legal option is applied by
the engine, so callers only ever see choices that matter. This matches the
online client's behaviour, which also keeps log replay aligned.

## Correctness

The rules are the foundation everything else rests on, and a wrong rule is
invisible at runtime — it just quietly trains the AI on a different game.

* **Fuzzing** (`core/tests/fuzz.rs`): 400 random-playout games on random
  kingdoms, plus 12 games for each of the 26 kingdom cards with that card forced
  into the supply. Asserts termination, that no decision is offered with fewer
  than two options, and that cards are conserved across supply, trash and all
  zones.
* **Targeted rule tests** (`core/tests/rules.rs`): 29 tests over the cards whose
  wording is easy to get subtly wrong — Merchant's per-copy Silver bonus, Throne
  Room, Moat reactions, Library's set-aside, Sentry's trash/discard/reorder,
  Bandit's victim-chooses trash, Vassal, Harbinger, Gardens scoring, and the
  full cost and type table.

That last one earned its place immediately: it caught Gold priced at $5 and
Duchy at $4, which had silently made every money bot buy Gold at $5 and made the
Witch strategy indistinguishable from plain Big Money.

## Hidden information and determinization

Dominion hides less than it looks like it does. Every card either player owns is
public — gains, trashes and discards all happen face up. What is hidden is only
*order*: which of my cards are on top of my deck, and which of the opponent's
unseen cards are in hand rather than in the deck.

So sampling a consistent world means reshuffling unknown order, never guessing
unknown contents, and perfect-information Monte Carlo is a much better
approximation here than in poker or bridge. `PlayerState::known_top` tracks
cards the owner deliberately put on top (Harbinger, Artisan, Sentry,
Bureaucrat), so determinization does not shuffle away the thing those cards are
for.

## Search: ISMCTS

`dominion-ai` runs Information Set MCTS with a PUCT selection rule: one shared
tree, re-determinized every iteration, so all samples accumulate into the same
statistics.

It did not start there. Plain UCT lost every game to BM+Smithy with near-uniform
visit counts — ending turns without playing Treasure and buying Curses. The
branching factor at a buy is ~15, adjacent buys differ by about a point of win
probability, and a rollout returns one bit, so the counts came out flat and the
search picked noise. Three fixes, all structural:

* **`prior::restrict`** drops provably bad moves. The load-bearing one is that
  playing a Treasure in the buy phase is never a mistake in the Base set (no
  Treasure has a downside, nothing cares about unspent money) and the order
  cannot matter, so the entire choice collapses to one forced move.
* **`prior::priors`** supplies a distribution over what is left. This is the
  slot the policy network fills; the search does not care where it comes from.
* **PUCT with first-play urgency** instead of expanding every child once before
  learning anything.

Then the tree itself. The first version was plain PIMC: N independent
determinized worlds, an independent tree in each, visit counts summed at the
end. At 8 worlds x 400 iterations every node was backed by at most 400 samples
though 3200 were paid for. ISMCTS shares one tree across all of them, treating a
node as an information set that holds the union of moves any world offered,
with **availability counts** so a move legal in a tenth of the worlds is not
judged against a universally-legal sibling on raw visits.

Dominion suits this unusually well: the deciding player knows their own hand, so
every determinization offers the same moves at the root, and move sets only
diverge deeper once cards have been drawn.

## Value targets: TD(lambda)

A game runs to roughly 240 decisions. Training every one of them against the
final win/loss is one bit of signal spread impossibly thin — a turn-three buy
credited with an outcome it barely influenced.

Self-play instead records a TD(lambda) target, walking each trajectory backwards
with `G_t = (1-λ)·V(s_{t+1}) + λ·G_{t+1}`, λ = 0.9. The bootstrap comes from
**the search**, not the network: the root value of a few thousand ISMCTS
simulations beats a single forward pass, so the targets are informative even
while the network is still weak.

Confirmed on identical data, differing only in the value target:

| | win rate | |
|---|---|---|
| TD net vs MC net | **64.67% ± 2.76%** | 5.3σ |

## The heuristic, and why it matters so much

`policy::gain_preference` is load-bearing three times over: it is the buy policy
of the baseline bots, the prior that steers the search, *and* the rollout policy
that assigns value to leaf positions. A version that only ever buys money makes
the search structurally blind to everything else, however many iterations it
runs.

Getting it right was empirical, and the ladder rather than intuition decided each
round:

| version | avg. win rate vs the ladder |
|---|---|
| money-only (buys no kingdom card) | 43.5% |
| buys a wide spread of engine pieces | 43.5% — *worse than BM+Smithy* |
| money with a few exceptions | 50.9% |
| + reserve the terminal slot for Witch | 61.1% |
| + allow the second Witch | **64.1%** |

The two interesting failures: a deck with one of everything draws none of it; and
a $4 Smithy bought on turn two consumes the deck's only terminal slot and locks
out the Witch the deck actually wanted, which is how a policy that ranked Witch
above Gold still lost 72% to Double Witch.

## Benchmark ladder

`cargo run --release --bin ladder -- 500` plays a round robin on random kingdoms
containing whatever cards both strategies need. Every matchup is seat-swapped on
paired seeds, so both agents see the same shuffles from both seats.

```
Heuristic       64.1%   (beats every menu below)
DoubleWitch     80.9%   \
BM+Smithy       60.4%    | among themselves
MilitiaMoney    54.6%    |
ChapelMoney     38.1%    |
BigMoney        34.1%    |
VillageSmithy   32.0%   /
```

The menu results match Dominion folklore, which is the point of having them:
cursing is the strongest single effect in the Base set, and BM+Smithy beats plain
Big Money around 70% of the time.

## Consulting the AI about a game in progress

The engine is deterministic given a seed and a move sequence, so a `GameLog` is
just `(kingdom, players, seed, moves)` — no recorded state. `replay_prefix(n)`
rebuilds the exact position after the first `n` moves, and `advise_log` searches
it:

```
$ cargo run --release --bin advise -- --demo
turn 5 — player 0 — BuyPhase
recommended: buy Gold (48% of visits)
considered:
  buy Gold     4655
  buy Silver   1525
  buy Sentry    766
  buy Merchant  661
```

The visit spread is the honest measure of how confident the search is.

## What did not work, and what that ruled out

Seven things were tried and measured. Six failed, and the failures are more
informative than the success — each one removed a plausible direction, and
together they converge on a single cause, described in the section after this
one.

| Change | Result | |
|---|---|---|
| TD(λ) value targets | 64.67% ± 2.76 vs MC | ✅ 5.3σ |
| Merge 4 machines' networks by averaging | 48.12% ± 2.50 vs the best single one | ❌ |
| Double the training data (twice) | 49.25% ± 3.54 | ❌ flat |
| 7x wider network (512×256) | 50.50% ± 2.89 | ❌ 0.2σ |
| Adam instead of SGD | loss 0.9046 → 0.8726, strength 50.63% ± 2.50 | ❌ 0.3σ |
| 16x the search budget | 50.42% ± 4.56 | ❌ 0.1σ |
| Flatten the prior to unlock the search | 27.50% ± 4.08 at 16x800 | ❌ 5.5σ *worse* |

**Weight merging.** Four contributors totalling 12,230 games averaged into a
network that could not beat the single 3,500-game one. Federated averaging
assumes each contributor takes a *small* step from a shared checkpoint. Eight
full epochs over hundreds of thousands of examples is not a small step, and two
networks that have travelled that far stop being averageable even from a common
start — hidden unit 7 in one no longer corresponds to hidden unit 7 in the
other. The "same starting checkpoint" precondition is necessary but not
sufficient.

**More data and more capacity both did nothing**, which pointed at the real
limit. Measuring the entropy of the policy targets found it: cross-entropy
cannot fall below the entropy of its target, the targets average **0.761 nats**
over the examples that carry any signal, and training sits at **0.903**. The
network already reproduces the search's policy to within ~0.14 nats. It is not
underfitting, and no amount of data or width changes that.

Two side findings from the same measurement:

* **61.7% of examples had exactly one legal move** after restriction, so their
  policy target is a point mass and their gradient is exactly zero. They cost
  that share of every epoch. They now skip the policy head — but they are *kept*
  for the value head, because dropping them outright cost 61.7% of its training
  data and measurably hurt it (value loss 0.0060 → 0.0088).
* The remaining ceiling is therefore **the quality of the search that produced
  the targets**. Imitating a mediocre search perfectly yields a mediocre player.
  Deeper search at generation time is the next thing to test; it costs 5.8x per
  game for 3.3x the iterations, so the honest comparison is compute-matched, not
  game-matched.

## The search reproduces its prior, and that turned out to be a good thing

Extra search bought nothing: 16x the budget won 50.42% ± 4.56 against 1x.
`search_agreement` found why — across a 16x budget range the search changed
the prior's top move in 0.7%–5.7% of decisions, and the top move's *visit
share rose* from 68.5% to 71.1%. The search was close to an identity function
over its own prior.

The cause was a feedback loop with a clear origin. `prior::priors` gave the
heuristic's move a weight of 6.0 against 1.0 for everything else; the search
therefore produced visit distributions with ~70% of the mass on one move; the
network trained on those distributions learned to reproduce that
concentration; and PUCT's `c·P·√N/(1+n)` then spent every additional
iteration reinforcing the move the prior already liked.

So prior temperature was added to flatten it, and `unlock_sweep` confirmed
the mechanism works — at temperature 4.0 the top move's visit share falls to
41.1% and 16x the budget changes the decision in 25.4% of positions instead
of 6.5%.

**It made the AI much weaker, and worse in proportion to how hard it
searched.**

| | result | |
|---|---|---|
| unlocked 16x800 vs unlocked 4x200 | 35.00% ± 4.35 | −108 Elo |
| unlocked 8x400 vs locked 8x400 | 38.33% ± 4.44 | −83 Elo |
| unlocked 16x800 vs locked 16x800 | 27.50% ± 4.08 | −168 Elo |

Same network on both sides throughout, so search configuration is the only
variable. Harm that scales with the budget is also the signature of a sign or
perspective bug in backup, so the next measurement checked for one directly —
`value_calibration` scores the raw value head, and the search's root value at
three budgets, against what actually happened:

| estimator | Brier | correlation |
|---|---|---|
| value head (raw) | 0.1691 | 0.521 |
| search 4x200 | 0.1668 | 0.536 |
| search 8x400 | 0.1663 | 0.537 |
| search 16x800 | 0.1660 | 0.537 |
| always 0.5 | 0.2500 | — |

No bug: the tree improves its own estimate, monotonically, in the right
direction. It improves it by 1.8% relative, saturating at 8x400 — 16x the
budget over 4x buys 0.0008 Brier.

That single table explains every failed result above. The search's `Q` is an
average of leaf values from an evaluator correlating 0.52 with the outcome;
averaging more of them reduces variance but not bias, so `Q` can never
resolve the difference between two candidate buys. The prior carries real
Dominion knowledge distilled from the heuristic. `Q`, at the resolution move
selection needs, is close to noise. Flattening the prior traded the knowledge
for the noise.

The prior was not a cage around a better search. It was the strongest
component in the system, and the search was riding it. **The binding
constraint is the accuracy of the leaf evaluation** — not the search, not the
prior, not the optimizer, and not the amount of data.

## The value head was never better than what it replaced

The value head exists to skip the rollout: an O(1) estimate instead of playing
the game out. That is a claim about *cost*. Whether it was also more
*accurate* was assumed and never measured — so `value_calibration` measured
it, running the identical search with the value head withheld so every leaf
falls back to a heuristic rollout.

| estimator | Brier | correlation |
|---|---|---|
| value head (raw) | 0.1636 | 0.547 |
| search 8x400, value head | 0.1599 | 0.572 |
| **search 8x400, rollout** | **0.1301** | **0.639** |

| Brier by third of game | early | middle | late |
|---|---|---|---|
| value head | 0.2027 | 0.1648 | 0.1212 |
| search 8x400 | 0.2004 | 0.1666 | 0.1104 |
| **rollout 8x400** | **0.1769** | **0.1287** | **0.0822** |

The rollout is 19% better on Brier, ahead in every third of the game, and
furthest ahead late — where the position is most decidable and being right
matters most. Every simulation the search has ever run was backing up a
leaf estimate worse than one that was available for free.

Better calibration is not automatically stronger play, and the rollout costs
**4.2x per game** (15.4s against 3.7s at 8x400, measured, not assumed). So
`leaf_showdown` ran it both ways, same network on both sides:

| | result | |
|---|---|---|
| rollout 8x400 vs value head 8x400 (game-matched) | 64.17% ± 4.38 | ✅ +101 Elo, 3.2σ |
| rollout 8x400 vs value head 8x1673 (equal wall clock) | 59.17% ± 4.49 | ✅ +64 Elo, 2.0σ |
| the same, re-run over 500 games | 57.00% ± 2.21 | ✅ +49 Elo, 3.2σ |

It wins even when the value head is handed 4.3x the simulations to spend the
same time. The wall-clock arm is the one that decides a default, so it was
re-run at 500 games before flipping anything; the point estimate settled from
59.17% to 57.00% and the significance rose from 2.0σ to 3.2σ. **The rollout
is now the default** (`MctsConfig::use_value_head` defaults to `false`). This is the second thing to work, and the largest single effect
found after TD(λ) — and it came from *removing* a component rather than
adding one.

### But it does not transfer to the training data

The obvious next inference — that a stronger search generates better training
targets, so all existing data is handicapped — was tested and is false.

Two corpora were generated over the **same kingdoms and the same seeds**,
4,907 games each, differing only in the leaf estimator. `corpus_diff`
confirms the play genuinely differs: 0.3% identical move sequences, first
divergence at move 17 (median), mean policy L1 0.274 over 65,347 shared
decisions. Two networks were then trained from the same checkpoint, on the
same number of games, with identical hyperparameters.

| | result | |
|---|---|---|
| net trained on rollout games vs net trained on value-head games | 49.50% ± 2.89 | ❌ 0.2σ |

No difference. A better search plays better without teaching better, and
rollout generation costs 4.3x. So the rollout is the right default *for
play*, and the wrong way to spend a self-play budget.

Two things this rules out. Regenerating the corpus is not worth doing, which
saved four machines a day of work on a measured null. And "the targets come
from a mediocre search" is no longer a sufficient explanation for why more
data measures flat — the search was made materially stronger and the trained
result did not move.

## Where it stands

`cargo run --release --example standing` measures the shipping configuration
against the opponents whose strength is known:

| | win rate | |
|---|---|---|
| vs the heuristic it uses as prior and rollout | 81.25% ± 3.56 | +255 Elo, 8.8σ |
| vs the strongest hand-written menu (Double Witch) | 78.33% ± 3.76 | +223 Elo, 7.5σ |
| vs the same search with no network | 61.58% ± 1.99 | +82 Elo, 5.8σ |

And against every menu on the benchmark ladder individually, at the cheaper
4x200 search (`cargo run --release --example vs_ladder`):

| opponent | win rate | |
|---|---|---|
| BigMoney | 92.92% ± 2.34 | 18.3σ |
| BM+Smithy | 80.00% ± 3.65 | 8.2σ |
| DoubleWitch | 77.08% ± 3.84 | 7.1σ |
| MilitiaMoney | 87.50% ± 3.02 | 12.4σ |
| ChapelMoney | 90.00% ± 2.74 | 14.6σ |
| VillageSmithy | 92.08% ± 2.46 | 17.1σ |
| **average** | **86.60%** | |

Against the 64.1% the hand-written heuristic averages on the same ladder.
Reported per menu rather than as one number because Big Money is a family, and
an average would hide a loss to one member of it — there isn't one, but that
was worth checking rather than assuming.

The third row is the one worth reading: it is what training bought, isolated
from the search that would run anyway. It is also a lesson in sample size.
The same matchup over 120 games measured 55.42% ± 4.54 — 1.2σ, which reads as
"the network cannot be shown to help at all" and is the wrong conclusion. Six
hundred games put it at 5.8σ. Both measurements are consistent; the first was
simply too noisy to act on, and acting on it anyway is the mistake this
project has made more than once.

**No number here is against the app's Hard bot.** Nothing in this project has
ever been measured against it, so the original target remains unquantified.

## Closing the self-play loop, and what eight nulls add up to

Every corpus in the project — all 24,000 games, from four machines — was
generated by a search guided by the *same* `models/net.bin`. Training had
therefore only ever seen generation-1 data, and the loop that makes
AlphaZero-style training work (train, regenerate with the **new** network,
retrain) had never actually been run. If a fixed data distribution were why
more data kept measuring flat, this is the experiment that would show it.

2,200 games generated by the newest network, trained from it, played against
it:

| | result | |
|---|---|---|
| generation 2 vs generation 1 | 51.00% ± 2.89 | ❌ 0.3σ |

Nothing. That makes eight training-side changes measured and eight nulls —
but the eight have something in common, and so do the two successes:

| change | what it altered | |
|---|---|---|
| TD(λ) value targets | what the network is taught | ✅ 64.67% |
| rollout leaf evaluation | what information enters the search | ✅ +101 Elo |
| data volume, network width, optimizer, data quality, data distribution | how much, or how well-processed | ❌ 8 of 8 |

**Both things that worked changed what information enters the system. All
eight failures changed how much of it there is, or how it is processed.** The
binding constraint is the information content of the targets, not capacity,
optimisation or volume — and there is a direct measurement of why: the policy
target is the search's visit distribution, and 16x the search budget changes
the chosen move in 5.7% of decisions. The network is largely being taught
what it already says.

## Does it refuse to build engines, or is money simply right?

The search prefers money on every kingdom and never assembles a thin-deck
engine. There is a structural reason it could not: Chapel costs money now and
repays five to eight turns later, the tree reaches two or three buys deep, and
everything past the tree is priced by a rollout that goes on buying money. The
evaluator cannot see an engine's payoff, so more search *reinforces* the
refusal rather than curing it.

That story is tidy enough to act on, which is why it was worth measuring
first. Engine-leaning menus, scored against the same six-menu ladder the
heuristic's 64.1% comes from:

| menu | avg vs the ladder |
|---|---|
| Chapel engine, no Gold | 12.75% |
| the same, with Gold | 46.00% |
| Laboratory + money | 38.89% |
| Festival + Laboratory | 36.22% |
| Village/Smithy engine, no Chapel | 19.69% |
| *the money-leaning heuristic* | *64.1%* |

The first row nearly produced the wrong conclusion. A menu without Gold
collapses to 12.75%, which reads as "engines are hopeless" and is really "this
menu cannot buy anything"; adding Gold moves the identical strategy to 46%.
The difference measured the menu, not Dominion.

**That conclusion was then withdrawn.** A human beat the AI with exactly the
Throne Room + Vassal line these numbers were taken to rule out, and checking
why turned up three separate defects, each of which on its own invalidates
the measurement:

1. **The menu did not contain the combo.** Adding Gold to the engine list also
   dropped Throne Room and Vassal, so the only variant that ever held them was
   the broken one with no economy. The 46% row is a Laboratory/Festival deck.
2. **The shared policy would not play it.** `throne_value` ranked Vassal in
   its lowest bucket, below every other Action, so a bot holding Throne Room
   and Vassal doubled almost anything else. It bought the pieces and refused
   to assemble them. Fixing that alone moved a Throne/Vassal menu from 5.5% to
   13.9%, and left the heuristic's own ladder average unchanged at 64.1%.
3. **The deck stays Copper-heavy.** Vassal only chains when the top of the
   deck is an Action, and `worth_trashing` stops thinning at about $4 of coin,
   so Vassal mostly just pays $2. Thinning harder in Action-dense decks helped
   thin engines (21% → 30%) and hurt the best variant (46% → 37%), so it was
   not kept — but it shows the trash policy, not the strategy, is deciding
   these numbers.

So nothing here measured whether engines are good in Base 2E. Every run
measured whether *these bots can execute one*, and they cannot. The claim that
money is stronger is unsupported, and the direct evidence — a human winning
with the line — stands.

It also explains the loss. The same policy is the buy heuristic, the search
prior *and* the rollout, so the AI inherits every one of these blind spots
three times over: it cannot value the combo, cannot build it, and cannot
recognise it coming.

The consequence for expansions is the opposite of the obvious one. The reason
to add cards is not to teach new combos — Base already contains Chapel,
Festival, Throne Room and Vassal, so the combos are present and unused. It is
that Base contains too few positions where building an engine is the *correct*
play, so there is nothing for the search to be rewarded for finding. Cards
that make engines correct often enough to learn from are a prerequisite for
engine play, not a bonus on top of it.

## Underpowered measurements, four times in one day

Every one of these was measured at 120 games, read as promising, and re-run
larger:

| measurement | at 120 games | re-run | |
|---|---|---|---|
| network vs no network | 55.42% ± 4.54 | **61.58% ± 1.99** (600) | rose |
| rollout vs value head, equal wall clock | 59.17% ± 4.49 | 57.00% ± 2.21 (500) | held |
| 4x search budget | 54.17% ± 4.55 | 52.00% ± 1.77 (800) | fell |
| 16x search budget | 57.92% ± 4.51 | 54.17% ± 2.88 (300) | fell |

Three of the four moved by more than a standard error, and two collapsed
into noise. A 120-game match has a standard error near 4.5%, which is wider
than every effect this project has found except two — so at that sample size
most results are unreadable, in either direction.

The search-scaling numbers are the substantive casualty. Fixing the leaf
estimate did improve how the search uses compute — 16x the budget went from
50.42% to 54.17% — but neither budget reaches significance, and 16x the
compute buying at most +29 Elo is nowhere near proportional. Deep search at
generation time stays ruled out on cost, and "the search largely reproduces
its prior" survives the better leaf estimate.

## Why per-card synergy weights cannot express an engine

The reason the AI never buys Throne Room or Vassal is not a preference for
money. It is a filter: anything not named explicitly in `gain_preference`
falls through to `is_action() => 300 + cost`, which is below Silver's 700, so
those cards are never bought at any price. A deck that cannot buy Throne Room
cannot be measured playing one.

The obvious fix is to price them by what the deck already holds — Throne Room
by how many good targets it owns, Vassal by Action density, Chapel by how much
junk is left to trash. Measured against the ladder the heuristic scores 64.14%
on:

| ranking | avg vs the ladder |
|---|---|
| baseline | 64.14% |
| + Vassal priced by Action density | 64.14% |
| + Throne Room priced by targets | 62.64% |
| + Chapel priced by junk | **50.12%** |
| all three | 49.09% |

Chapel alone costs fourteen points. That is not a tuning failure, it is the
shape of the problem: buying Chapel commits the deck to being thin, the trash
policy duly strips it to about $4 of coin, and then the buy policy goes on
buying money with a wrecked economy. Thinning without an engine to thin *for*
is simply worse than not thinning. Vassal's rule never fires at all — its
Action-density condition is never met by a money deck, which is the same point
from the other side.

**A Dominion strategy is a package, and a per-card ranking function cannot
represent one.** Every card in an engine is a bad buy until the others are
present, so no ordering of independent per-card scores reaches the engine:
each first step is locally wrong. This is the same wall the "wide spread of
engine pieces" version hit at 43.5%, now with the mechanism identified rather
than just the symptom.

What that implies for the next attempt: the buy policy needs to commit to a
plan the kingdom supports and then buy consistently for it, rather than score
cards one at a time. That is a different shape of function, not a better set
of weights.

## How much is plan commitment worth? About a point.

Since per-card weights cannot express an engine, the natural next step was a
buy policy that commits to a plan the kingdom supports. That is a large piece
of work, so the headroom was measured before building any of it.

`plan_headroom` plays every hand-written plan a kingdom actually supports
against the heuristic, on that kingdom, and takes the best per kingdom **with
hindsight**. That oracle is an upper bound on what perfect plan selection
could ever buy. Over 80 random kingdoms, 120 games per plan per kingdom:

| | |
|---|---|
| oracle: best supported plan per kingdom | **51.05%** |
| the heuristic (it is the opponent) | 50% by construction |

| plan | avg vs heuristic | kingdoms it won |
|---|---|---|
| BM+Smithy | 51.15% | 23 |
| DoubleWitch | 45.94% | 27 |
| MilitiaMoney | 36.15% | 8 |
| ChapelMoney | 35.38% | 1 |
| Lab+Money | 34.40% | 2 |
| BigMoney | 30.54% | 19 |
| VillageSmithy | 23.37% | 0 |
| Throne/Vassal | 15.83% | 0 |

Perfect hindsight plan selection is worth **one point**. Whatever machinery
would choose the plan, it cannot beat the oracle, so plan selection is not
worth building — the generic ranking already plays about as well as the best
of nine hand-written plans chosen per kingdom after the fact.

A second finding fell out of it. `ChapelEngine` never appears: it needs six
specific kingdom cards, and a random 10-of-26 kingdom contains all six about
0.09% of the time. In 80 kingdoms it was never available to play. **An engine
that needs six pieces is almost never on the table** — which is a structural
argument about how often engine play is even an option, separate from how
strong it is when it is.

The caveat that applies to all of it: an oracle is only as good as its plan
library, and this library is weakest exactly where engines live. Throne/Vassal
scores 15.83% while being crippled by a trash policy that stops thinning at
$4 and a priority list that cannot buy conditionally. So this rules out plan
selection over *these* plans; it does not rule out that a competently
expressed engine plan exists and would change the table.

## The AI has never once tried to build an engine

"Prefers money" is a claim about a mechanism, so it is worth checking as a
fact. `buy_profile` puts every engine piece in the supply — Chapel, Festival,
Throne Room, Vassal, Village, Laboratory — gives the AI its normal 8x400
search, and counts what it buys over 30 games.

| card | bought | offered | taken |
|---|---|---|---|
| Province | 113 | 136 | 83.1% |
| Gold | 112 | 361 | 31.0% |
| Silver | 83 | 1302 | 6.4% |
| Militia | 48 | 879 | 5.5% |
| Laboratory | 35 | 569 | 6.2% |
| Market | 25 | 569 | 4.4% |
| Vassal | 3 | 1302 | 0.2% |
| **Chapel** | **0** | 1772 | **0.0%** |
| **Festival** | **0** | 569 | **0.0%** |
| **Throne Room** | **0** | 879 | **0.0%** |
| **Village** | **0** | 1302 | **0.0%** |

Four cards were offered roughly 4,500 times and bought zero times. Vassal was
taken three times out of 1,302.

The split is exactly what the code predicts. Laboratory, Market and Militia
are named in `gain_preference` and get bought. Chapel, Festival, Throne Room
and Village fall through to `is_action() => 300 + cost`, below Silver's 700,
and so lose every comparison *before* the search begins — the prior never
offers them any weight, and the search allocates its budget by the prior.

This changes what the earlier results can be read to mean. The AI has never
chosen money over an engine, because it has never once tried an engine. So
"money is the right call for it" is untested from its side as well: there is
no side. Whatever else is true about Base 2E, a human who builds an engine
against this AI is playing in a lane it does not contest.

## Sharing self-play between machines

Self-play parallelises across machines; the data did not. A 3000-game `.shard`
of expanded feature vectors is 427MB, past what GitHub accepts, which is what
pushed us toward merging networks instead — and that failed.

The fix is that the features are *derivable*. The engine is deterministic given
a kingdom, a seed and a move list, so the compact `.gamelog` format stores the
game and re-derives features by replaying at training time: 30 bytes per
decision instead of 600, **23MB per 3000 games instead of 427MB**. Replay costs
seconds against the hours the games took to search.

Both formats are readable, so no earlier data is stranded, and both are written
incrementally — a run killed at any moment leaves a valid file, and the reader
stops at a half-written trailing record rather than rejecting the whole thing.
That was verified by SIGKILLing a live run and recovering 4283 intact examples,
not just by unit test.

## Status

- [x] Base 2E engine, fuzzed and unit-tested
- [x] Determinization with topdeck knowledge preserved
- [x] ISMCTS with priors and availability counts
- [x] TD(λ) value targets, confirmed at 5.3σ
- [x] Compact shareable self-play format
- [x] Game logs, prefix replay, advice API
- [x] ~~Deeper search at generation time~~ — ruled out, twice. 16x the budget
      is worth 50.42% ± 4.56, and the calibration table above says why: the
      tree improves its leaf estimate by 1.8%, saturating at 8x400.
- [x] A leaf evaluation worth searching with — the value head was measurably
      worse than the rollout it replaced, and swapping it out is worth
      +101 Elo game-matched and +64 Elo at equal wall clock.
- [x] ~~Regenerate self-play with rollout leaves~~ — tested on matched
      corpora and measured null (49.50% ± 2.89). Stronger search, same
      trained strength.
- [ ] **Why does better data not help?** Six directions have now failed and
      the two that worked both act at play time, not training time. The
      policy head is already within 0.14 nats of its target entropy, so it is
      not underfitting — the targets themselves may be the limit, and the
      visit distribution is dominated by a prior the search barely overrides.
- [ ] **Statistical power.** Results land at ±2.5–4.5% on 120–500 games, and
      two directions were pursued on numbers that later moved by 2σ when
      re-run larger. Parallel machines are better spent here than on more
      self-play.
- [ ] Faster training: the loop is single-threaded scalar Rust, and a 512×256
      network takes 45 minutes for 6 epochs. Batching and SIMD are worth
      10-50x and now gate iteration speed.
- [ ] Parser for real Dominion Online logs

## Measuring against the Hard bot

The official app is closed source, so there is no way to run automated matches
against its Hard bot. The plan is to use the open ladder plus self-play Elo as
the fast automated signal, and to validate against Hard manually over a few dozen
games at the end. If the gap is as large as it should be, a few dozen games is
enough to see it.

## Where to run training

Dominion self-play is a CPU workload: the network is a small MLP and the
bottleneck is generating games. Cores matter, GPUs do not, which makes free GPU
notebooks a poor fit — they are generous with GPU and stingy with vCPU.

Measured: ~0.23 games/s on 4 cores at 8x300 iterations, ~0.04 games/s at 8x1000.
`bin/bench` prints the same figures for any machine.

**Buying a bigger machine is not currently justified.** The learning curve is
flat — doubling the data twice changed nothing measurable — so a server that
generates 10x more games of the same kind buys 10x more of something that has
stopped helping. That calculus changes only if deeper search turns out to help,
since deeper search is what a bigger machine would actually be for.
