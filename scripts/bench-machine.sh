#!/usr/bin/env bash
# Benchmark turbosort on this machine and print a summary to paste back.
# Meant for a fresh Ubuntu server, run inside tmux:
#
#   curl -sSfLO https://raw.githubusercontent.com/ampactor-labs/turbosort/main/scripts/bench-machine.sh
#   bash bench-machine.sh              # main, the full set (~1.5-2.5 h)
#   bash bench-machine.sh v0.2.2 quick # another ref; one headline run only
#
# Runs the README's headline groups (u32/u64/f32/i32 against std and
# voracious, and the parallel sort) three times, then the small-slice,
# patterned and sort_by_key groups once, then sort_parallel at 10M across
# thread counts, and a perf profile if the machine exposes hardware
# counters. Single-threaded groups are pinned to one NUMA node on
# multi-socket machines. Writes everything to ~/turbosort-bench-results
# and ~/turbosort-bench-results.tar.gz.
set -euo pipefail

REF="${1:-main}"
MODE="${2:-full}"
OUT="$HOME/turbosort-bench-results"
SRC="$HOME/turbosort-bench-src"
mkdir -p "$OUT"

# Toolchain: a C linker, git, python3 and a stable Rust.
if ! command -v cc >/dev/null || ! command -v git >/dev/null || ! command -v python3 >/dev/null; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq build-essential git python3 curl >/dev/null
fi
if ! command -v numactl >/dev/null; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get install -y -qq numactl >/dev/null 2>&1 || true
fi
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true

if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/ampactor-labs/turbosort "$SRC"
fi
git -C "$SRC" fetch -q --tags origin
git -C "$SRC" checkout -q --detach "$REF"

{
  echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "ref: $REF ($(git -C "$SRC" rev-parse --short HEAD))"
  echo "rustc: $(rustc -V)"
  echo "kernel: $(uname -sr)"
  echo "nproc: $(nproc)"
  lscpu | grep -E '^(Model name|Thread|Core|Socket|CPU max MHz|L1d|L2|L3|Hypervisor vendor|NUMA node\(s\))' || true
  echo "avx2: $(grep -c -w avx2 /proc/cpuinfo) of $(nproc) cpus; avx512f: $(grep -c -w avx512f /proc/cpuinfo)"
  echo "memory: $(free -g | awk '/^Mem/ {print $2 " GiB"}')"
  echo "pmu: $([ -d /sys/bus/event_source/devices/cpu ] && echo yes || echo no)"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$OUT/machine.txt"
cat "$OUT/machine.txt"

# One NUMA node for single-threaded groups, so memory stays local.
PIN=()
if command -v numactl >/dev/null && [ "$(numactl --hardware | awk '/^available:/ {print $2}')" -gt 1 ]; then
  PIN=(numactl --cpunodebind=0 --membind=0)
  echo "numa: pinned single-threaded groups to node 0" >> "$OUT/machine.txt"
fi
(cd "$SRC" && cargo bench -q --bench sort_bench --features parallel --no-run 2>"$OUT/build.log") \
  || { echo "!! build failed; see $OUT/build.log" >&2; tail -20 "$OUT/build.log" >&2; exit 1; }

bench() { # name, criterion filter, extra args...
  local name="$1" filter="$2"; shift 2
  echo ">> $name ($(date -u +%H:%M))"
  local pin=("${PIN[@]}"); [[ "$filter" == *parallel* ]] && pin=()
  (cd "$SRC" && "${pin[@]}" cargo bench -q --bench sort_bench --features parallel -- "$filter" \
      --noplot --output-format bencher "$@" 2>"$OUT/$name.log") | grep '^test' > "$OUT/$name.txt" \
    || { echo "!! $name failed; see $OUT/$name.log" >&2; tail -20 "$OUT/$name.log" >&2; exit 1; }
}

runs=3; [ "$MODE" = quick ] && runs=1
for i in $(seq 1 "$runs"); do
  bench "headline_$i" '^(turbosort|parallel)/' --sample-size 30
done
if [ "$MODE" != quick ]; then
  bench small_batches '^small_batches/' --sample-size 20 --warm-up-time 1 --measurement-time 3
  bench patterns '^patterns/' --sample-size 20 --warm-up-time 1 --measurement-time 2
  bench by_key '^by_key/' --sample-size 10 --warm-up-time 1 --measurement-time 3

  # Against Google Highway's vqsort, when the ref has the comparison program
  # and Highway's library installs.
  if [ -f "$SRC/compare/Cargo.toml" ]; then
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    $SUDO apt-get install -y -qq libhwy-dev pkg-config >/dev/null 2>&1 || true
    echo ">> compare ($(date -u +%H:%M))"
    (cd "$SRC/compare" && "${PIN[@]}" cargo run -q --release > "$OUT/compare.md" 2>"$OUT/compare.log") \
      || echo "!! compare failed; see $OUT/compare.log" >&2
  fi

  # sort_parallel at 10M across thread counts, doubling up to every CPU.
  t=1
  while :; do
    RAYON_NUM_THREADS=$t bench "threads_$t" '^parallel/u32/turbosort_par/10000000$' --sample-size 10
    [ "$t" -ge "$(nproc)" ] && break
    t=$(( t * 2 )); [ "$t" -gt "$(nproc)" ] && t=$(nproc)
  done

  # Hardware counters, if the hypervisor passes them through.
  if [ -d /sys/bus/event_source/devices/cpu ]; then
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    command -v perf >/dev/null || $SUDO apt-get install -y -qq "linux-tools-$(uname -r)" linux-tools-common >/dev/null 2>&1 || true
    if command -v perf >/dev/null; then
      echo ">> perf ($(date -u +%H:%M))"
      (cd "$SRC" && cargo build -q --profile profiling --example profile 2>>"$OUT/build.log")
      for algo in ts std; do
        "${PIN[@]}" perf stat -e cycles,instructions,cache-references,cache-misses,dTLB-load-misses \
          -o "$OUT/perf_u32_$algo.txt" "$SRC/target/profiling/examples/profile" u32 10000000 30 "$algo" \
          > "$OUT/perf_u32_$algo.out" 2>&1 || true
      done
    fi
  fi
fi

python3 - "$OUT" > "$OUT/summary.md" <<'EOF'
import glob, os, re, statistics, sys
out = sys.argv[1]
line = re.compile(r"^test (\S+) \.\.\. bench:\s+([\d,]+) ns/iter")

def load(path):
    res = {}
    for l in open(path):
        m = line.match(l)
        if m:
            res[m.group(1)] = int(m.group(2).replace(",", ""))
    return res

def fmt(ns):
    for unit, div in (("s", 1e9), ("ms", 1e6), ("µs", 1e3)):
        if ns >= div:
            return f"{ns / div:.3g} {unit}"
    return f"{ns:.3g} ns"

print(open(os.path.join(out, "machine.txt")).read())
runs = [load(p) for p in sorted(glob.glob(os.path.join(out, "headline_*.txt")))]
print(f"## Headline groups, median of {len(runs)} run(s); the range is min-max of the per-run ratio\n")
print("| Group | Size | std | turbosort | voracious | vs std | vs voracious |")
print("| --- | --- | --- | --- | --- | --- | --- |")
keys = sorted({k for r in runs for k in r if k.startswith("turbosort/") and "/turbosort/" in k},
              key=lambda k: (k.split("/")[1], int(k.split("/")[-1])))
def cell(base, alg):
    ratios = [r[base] / r[alg] for r in runs if base in r and alg in r]
    if not ratios:
        return "-"
    return f"{statistics.median(ratios):.2f}x ({min(ratios):.2f}-{max(ratios):.2f})" if len(ratios) > 1 else f"{ratios[0]:.2f}x"
for k in keys:
    _, ty, _, size = k.split("/")
    std, vor = f"turbosort/{ty}/std_unstable/{size}", f"turbosort/{ty}/voracious/{size}"
    med = lambda n: fmt(statistics.median([r[n] for r in runs if n in r])) if any(n in r for r in runs) else "-"
    print(f"| {ty} | {size} | {med(std)} | {med(k)} | {med(vor)} | {cell(std, k)} | {cell(vor, k)} |")
par = sorted({k.split("/")[-1] for r in runs for k in r if k.startswith("parallel/")}, key=int)
if par:
    print("\n| Parallel u32 | std | serial | parallel | parallel vs std | parallel vs serial |")
    print("| --- | --- | --- | --- | --- | --- |")
    for s in par:
        p, q, d = (f"parallel/u32/{n}/{s}" for n in ("turbosort_par", "turbosort_seq", "std_unstable"))
        med = lambda n: fmt(statistics.median([r[n] for r in runs if n in r]))
        print(f"| {s} | {med(d)} | {med(q)} | {med(p)} | {cell(d, p)} | {cell(q, p)} |")

sweep = sorted(glob.glob(os.path.join(out, "threads_*.txt")), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
if sweep:
    print("\n## sort_parallel, 10M random u32, by thread count (one run each)\n")
    print("| Threads | Time | Speedup over 1 thread |")
    print("| --- | --- | --- |")
    one = None
    for p in sweep:
        t = int(re.findall(r"\d+", os.path.basename(p))[0])
        v = next(iter(load(p).values()), None)
        if v is None:
            continue
        one = one or v
        print(f"| {t} | {fmt(v)} | {one / v:.2f}x |")

cmp = os.path.join(out, "compare.md")
if os.path.exists(cmp):
    print("\n# Against vqsort (compare/)")
    print(open(cmp).read())

for p in sorted(glob.glob(os.path.join(out, "perf_*.txt"))):
    print(f"\n## {os.path.basename(p)}\n\n```")
    print("".join(l for l in open(p) if l.strip() and not l.startswith("#")).rstrip())
    print("```")

for name, base, algs in (("small_batches", "std_unstable", ["turbosort"]),
                         ("patterns", "std_unstable", ["turbosort"]),
                         ("by_key", "std_stable", ["turbosort", "turbosort_buffer"])):
    path = os.path.join(out, f"{name}.txt")
    if not os.path.exists(path):
        continue
    r = load(path)
    print(f"\n## {name} (one run), speed against {base}\n")
    rows = sorted({tuple(k.split("/")[1:]) for k in r if k.split("/")[1] == base})
    print("| Case | " + " | ".join(algs) + " |")
    print("| --- |" + " --- |" * len(algs))
    for row in rows:
        rest = "/".join(row[1:])
        b = r.get(f"{name}/{base}/{rest}")
        cells = [f"{b / r[f'{name}/{a}/{rest}']:.2f}x" if f"{name}/{a}/{rest}" in r else "-" for a in algs]
        print(f"| {rest} | " + " | ".join(cells) + " |")
EOF

tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo
echo "================ paste everything below this line ================"
cat "$OUT/summary.md"
echo "Raw results: $OUT.tar.gz"
