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
cargo test                                  # 45 tests: rules, fuzzing, logs, search
cargo run --release --bin ladder            # heuristic round robin
cargo run --release --bin bench             # machine check + search strength
cargo run --release --bin advise -- --demo  # ask the AI about a position
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

## Search

`dominion-ai` runs determinized PUCT: sample worlds consistent with the player's
view, search each, pick the move the ensemble visited most.

Plain UCT did not work at all — it lost every game to BM+Smithy with near-uniform
visit counts, ending turns without playing Treasure and buying Curses. The
branching factor at a buy is ~15, adjacent buys differ by about a point of win
probability, and a rollout returns one bit; the visit counts came out flat and
the search picked noise. Three fixes, all structural:

* **`prior::restrict`** drops provably bad moves. The load-bearing one is that
  playing a Treasure in the buy phase is never a mistake in the Base set (no
  Treasure has a downside, nothing cares about unspent money) and the order
  cannot matter, so the entire choice collapses to one forced move. That removes
  most of the buy-phase branching on its own.
* **`prior::priors`** supplies a distribution over what is left, concentrated on
  the heuristic's own pick. This is the slot a policy network will fill; the
  search does not care where the distribution comes from.
* **PUCT with first-play urgency** — `Q + c·P·√N/(1+n)` — instead of expanding
  every child once before learning anything.

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

## Status and roadmap

Done:

- [x] Complete Base 2E rules engine, fuzzed and unit-tested
- [x] Determinization with topdeck knowledge preserved
- [x] Heuristic policy calibrated against the ladder
- [x] Determinized PUCT search with priors
- [x] Game logs, prefix replay, and the advice API
- [x] Seat-swapped, seed-paired, multi-threaded match harness

Next, in order:

- [ ] **Value network.** The search's weakest link is that it evaluates a
      position by rolling it out with the heuristic — so it can only see what the
      heuristic can see. A learned value head replaces the rollout and is where
      the large gains are.
- [ ] **Policy network** to replace `prior::priors`, trained on search visit
      counts. The interface for this already exists.
- [ ] **Self-play loop** with Elo tracking against the ladder.
- [ ] Parser for real Dominion Online logs, so games against Hard can be
      analysed directly rather than transcribed.

## Measuring against the Hard bot

The official app is closed source, so there is no way to run automated matches
against its Hard bot. The plan is to use the open ladder plus self-play Elo as
the fast automated signal, and to validate against Hard manually over a few dozen
games at the end. If the gap is as large as it should be, a few dozen games is
enough to see it.

## Where to run training

Dominion is a CPU workload, not a GPU one: the network is a small MLP over a few
hundred features, and the bottleneck is generating self-play games. Cores matter,
GPUs mostly do not — which makes the free GPU notebook services a poor fit, since
they are generous with GPU and stingy with vCPU.

Measured here (4 vCPU): ~7,500 heuristic games/s/core, and ~0.6 searched games/s
across all four cores at 8 worlds × 400 iterations. `bin/bench` prints the same
figures for any machine.
