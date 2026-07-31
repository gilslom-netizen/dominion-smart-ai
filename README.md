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
| `dominion-core` | The rules engine. No dependencies. |
| `dominion-bots` | Heuristic reference agents, buy menus, match harness. |

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
invisible at runtime — it just quietly trains the AI on a different game. Two
layers of testing:

* **Fuzzing** (`tests/fuzz.rs`): 400 random-playout games on random kingdoms,
  plus 12 games for each of the 26 kingdom cards with that card forced into the
  supply. Asserts termination, that no decision is offered with fewer than two
  options, and that cards are conserved across supply, trash and all zones.
* **Targeted rule tests** (`tests/rules.rs`): 29 tests over the cards whose
  wording is easy to get subtly wrong — Merchant's per-copy Silver bonus, Throne
  Room, Moat reactions, Library's set-aside, Sentry's trash/discard/reorder,
  Bandit's victim-chooses trash, Vassal, Harbinger, Gardens scoring, and the
  full cost and type table.

That last one earned its place immediately: it caught Gold priced at $5 and
Duchy at $4, a bug that had silently made every money bot buy Gold at $5 and
made the Witch strategy indistinguishable from plain Big Money.

## Benchmark ladder

`cargo run --release --bin ladder -- 500` plays a round robin. Pairings use
random kingdoms containing whatever cards both strategies need, and every
matchup is played seat-swapped on paired seeds, so both agents see the same
shuffles from both seats.

Current standings (1000 games per pairing):

```
DoubleWitch     81.2%
BM+Smithy       60.1%
MilitiaMoney    54.5%
ChapelMoney     38.0%
BigMoney        34.2%
VillageSmithy   32.0%
```

These match Dominion folklore, which is the point of having them: cursing is the
strongest single effect in the Base set, and BM+Smithy beats plain Big Money
around 70% of the time. Throughput is ~14,000 full games/second on one core.

## Status and roadmap

Done:

- [x] Complete Base 2E rules engine, fuzzed and unit-tested
- [x] Heuristic play policy shared by all bots
- [x] Buy-menu strategies and the seat-swapped, seed-paired match harness

Next:

- [ ] Observation layer: per-player view with hidden information tracked as
      constraints, plus determinization (sampling a concrete hidden state) —
      needed both for honest self-play and for replaying real game logs
- [ ] MCTS over determinized states, with the heuristic policy as the rollout
      policy
- [ ] Self-play reinforcement learning: policy/value network over the state
      encoding, trained against the ladder and by self-play Elo
- [ ] A decision API that takes a game history and returns a recommended move,
      so the AI can be consulted mid-game without playing the whole game

## Measuring against the Hard bot

The official app is closed source, so there is no way to run automated matches
against its Hard bot. The plan is to treat the open ladder above plus self-play
Elo as the fast, automated signal, and to validate against Hard manually over a
few dozen games at the end. If the gap is as large as it should be, a few dozen
games is enough to see it.
