#!/usr/bin/env bash
# demo-hot-shard.sh — operator walkthrough for `--hot-shard-rps`.
#
# Spins up a 1-router + 2-serving + 1-Mode-B-spare topology, drives
# `--concurrent 16` traffic for ~30 s, and prints the rebalancer's
# elevation + cool-down lines so you can see the spare get pulled and
# returned.
#
# Companion to crates/larql-router/docs/hot-shard-demo.md.
#
# Usage:
#   VINDEX=path/to/your.vindex ./scripts/demo-hot-shard.sh
#
# Env knobs:
#   VINDEX                 — vindex path (required)
#   LARQL_BIN              — path to larql CLI                  (default: ./target/release/larql)
#   LARQL_ROUTER_BIN       — path to larql-router               (default: ./target/release/larql-router)
#   LARQL_SERVER_BIN       — path to larql-server               (default: ./target/release/larql-server)
#   HOT_SHARD_RPS          — threshold passed to --hot-shard-rps (default: 5)
#   REBAL_INTERVAL_SECS    — --rebalance-interval                (default: 5)
#   CONCURRENT             — bench --concurrent                  (default: 16)
#   TOKENS                 — bench --tokens                      (default: 50)
#   DEMO_LOG_DIR           — where to write router/server logs   (default: /tmp/hot-shard-demo)

set -euo pipefail

if [ -z "${VINDEX:-}" ]; then
    echo "error: set VINDEX=path/to/your.vindex" >&2
    exit 1
fi
if [ ! -d "$VINDEX" ]; then
    echo "error: VINDEX path does not exist or is not a directory: $VINDEX" >&2
    exit 1
fi

LARQL_BIN="${LARQL_BIN:-./target/release/larql}"
LARQL_ROUTER_BIN="${LARQL_ROUTER_BIN:-./target/release/larql-router}"
LARQL_SERVER_BIN="${LARQL_SERVER_BIN:-./target/release/larql-server}"
HOT_SHARD_RPS="${HOT_SHARD_RPS:-5}"
REBAL_INTERVAL_SECS="${REBAL_INTERVAL_SECS:-5}"
CONCURRENT="${CONCURRENT:-16}"
TOKENS="${TOKENS:-50}"
DEMO_LOG_DIR="${DEMO_LOG_DIR:-/tmp/hot-shard-demo}"

# Sanity-check the binaries exist before kicking anything off.
for bin in "$LARQL_BIN" "$LARQL_ROUTER_BIN" "$LARQL_SERVER_BIN"; do
    if [ ! -x "$bin" ]; then
        echo "error: $bin not found or not executable" >&2
        echo "  hint: cargo build --release -p larql-cli -p larql-router -p larql-server" >&2
        exit 1
    fi
done

mkdir -p "$DEMO_LOG_DIR"
ROUTER_LOG="$DEMO_LOG_DIR/router.log"
SHARD_A_LOG="$DEMO_LOG_DIR/shard-a.log"
SHARD_B_LOG="$DEMO_LOG_DIR/shard-b.log"
SHARD_C_LOG="$DEMO_LOG_DIR/shard-c.log"

PIDS=()
cleanup() {
    set +e
    echo
    echo "[cleanup] stopping ${#PIDS[@]} background processes…"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Router ─────────────────────────────────────────────────────────────
echo "[1/5] starting router on :50052 (gRPC) + :9090 (HTTP)…"
RUST_LOG=info "$LARQL_ROUTER_BIN" \
    --grid-port 50052 --port 9090 \
    --target-replicas 1 \
    --hot-shard-rps "$HOT_SHARD_RPS" \
    --rebalance-interval "$REBAL_INTERVAL_SECS" \
    > "$ROUTER_LOG" 2>&1 &
PIDS+=($!)
sleep 1

# ── Two serving shards (Mode A) ────────────────────────────────────────
for spec in "9181:a" "9182:b"; do
    PORT="${spec%:*}"
    NAME="${spec#*:}"
    LOG_VAR="SHARD_$(echo "$NAME" | tr a-z A-Z)_LOG"
    LOG_PATH="${!LOG_VAR}"
    echo "[2/5] starting shard-$NAME on :$PORT (Mode A, layers 0-9)…"
    RUST_LOG=info "$LARQL_SERVER_BIN" "$VINDEX" \
        --port "$PORT" --layers 0-9 \
        --join http://localhost:50052 \
        --public-url "http://localhost:$PORT" \
        > "$LOG_PATH" 2>&1 &
    PIDS+=($!)
done
sleep 2

# ── Mode B spare ───────────────────────────────────────────────────────
echo "[3/5] starting shard-c on :9183 (Mode B available, 8GB RAM)…"
RUST_LOG=info "$LARQL_SERVER_BIN" \
    --port 9183 \
    --available-ram 8GB \
    --join http://localhost:50052 \
    --public-url http://localhost:9183 \
    > "$SHARD_C_LOG" 2>&1 &
PIDS+=($!)
sleep 2

# ── Drive load ─────────────────────────────────────────────────────────
echo "[4/5] driving --concurrent $CONCURRENT × --tokens $TOKENS for ~30s to exceed $HOT_SHARD_RPS req/s…"
"$LARQL_BIN" bench "$VINDEX" \
    --backends "" \
    --ffn http://localhost:9090 \
    --concurrent "$CONCURRENT" \
    --tokens "$TOKENS" \
    > "$DEMO_LOG_DIR/bench.log" 2>&1 || true

# Let one or two rebalancer ticks fire after the load stops so we see
# the cool-down too.
echo "[5/5] waiting $((REBAL_INTERVAL_SECS * 3))s for cool-down ticks…"
sleep $((REBAL_INTERVAL_SECS * 3))

# ── Report ─────────────────────────────────────────────────────────────
echo
echo "──── Rebalancer events from $ROUTER_LOG ────"
grep -E "hot shard detected|hot shard cooled|Mode B assignment sent|dropping over-replicated" "$ROUTER_LOG" || \
    echo "(no rebalancer events fired — try a lower HOT_SHARD_RPS or higher CONCURRENT)"

echo
echo "Logs: $DEMO_LOG_DIR/"
echo "Done."
