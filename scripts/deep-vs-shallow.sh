#!/usr/bin/env bash
# Does a deeper search at generation time produce better training targets?
#
# The measurement this project keeps getting wrong is the controlled one, so
# this script exists to make the controls explicit rather than remembered:
#
#   * Both networks train with IDENTICAL hyperparameters. An earlier comparison
#     silently pitted a 6-epoch network against an 8-epoch one and measured the
#     epoch count.
#   * The comparison is a direct head-to-head. Measuring each side against a
#     common third party and comparing the two win rates carries independent
#     noise from both, and has twice been too weak to resolve a real effect.
#   * Generation cost is matched, not game count. Deep search runs 5.8x slower
#     per game, so "same number of games" would just be handing deep search 5.8x
#     the compute.
#
# One confound cannot be removed and is reported rather than hidden: at matched
# compute, deep search yields far fewer games, so data *quality* and data
# *quantity* move in opposite directions. The script therefore runs the
# comparison twice — once at matched epochs (each corpus trained the way you
# would normally train it) and once at matched gradient steps (same optimisation
# budget on both). Agreement means the conclusion is robust to that choice;
# disagreement means the effect is entangled with corpus size and neither number
# should be quoted alone.
#
# Usage: scripts/deep-vs-shallow.sh <deep-data-dir> <shallow-data-dir>

set -euo pipefail

DEEP=${1:?usage: deep-vs-shallow.sh <deep-dir> <shallow-dir>}
SHALLOW=${2:?usage: deep-vs-shallow.sh <deep-dir> <shallow-dir>}
EPOCHS=${EPOCHS:-6}
LR=${LR:-0.01}
PAIRS=${PAIRS:-150}
OUT=${OUT:-/tmp/deep-vs-shallow}

mkdir -p "$OUT"
cargo build --release --workspace 2>&1 | tail -1

echo "=== corpus sizes ==="
DEEP_N=$(cargo run --release --bin train -- --data "$DEEP"    --epochs 0 --eval-games 0 --net-out /dev/null 2>/dev/null | grep -oE '^[0-9]+ training examples' | grep -oE '^[0-9]+')
SHAL_N=$(cargo run --release --bin train -- --data "$SHALLOW" --epochs 0 --eval-games 0 --net-out /dev/null 2>/dev/null | grep -oE '^[0-9]+ training examples' | grep -oE '^[0-9]+')
echo "deep:    $DEEP_N examples"
echo "shallow: $SHAL_N examples"

echo
echo "=== run 1: matched epochs ($EPOCHS each) ==="
cargo run --release --bin train -- --data "$DEEP"    --epochs "$EPOCHS" --lr "$LR" \
    --net-out "$OUT/deep-epochs.bin"    --eval-games 0 2>/dev/null | grep -E "epoch|examples"
cargo run --release --bin train -- --data "$SHALLOW" --epochs "$EPOCHS" --lr "$LR" \
    --net-out "$OUT/shallow-epochs.bin" --eval-games 0 2>/dev/null | grep -E "epoch|examples"
cargo run --release --example net_vs_net -- "$OUT/deep-epochs.bin" "$OUT/shallow-epochs.bin" "$PAIRS"

echo
echo "=== run 2: matched gradient steps ==="
# Same total steps on both sides: cap the larger corpus, and give the smaller
# one enough epochs to match.
STEPS=$((DEEP_N * EPOCHS))
SHAL_EPOCHS=$EPOCHS
echo "target: $STEPS steps per side"
cargo run --release --bin train -- --data "$SHALLOW" --limit "$DEEP_N" --epochs "$SHAL_EPOCHS" --lr "$LR" \
    --net-out "$OUT/shallow-steps.bin" --eval-games 0 2>/dev/null | grep -E "epoch|examples|limited"
cargo run --release --example net_vs_net -- "$OUT/deep-epochs.bin" "$OUT/shallow-steps.bin" "$PAIRS"

echo
echo "Both runs above are deep-vs-shallow, deep listed first."
echo "Above 50% means deeper search produced better targets."
echo "If the two runs disagree, the effect is entangled with corpus size."
