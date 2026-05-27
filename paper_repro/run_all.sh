#!/usr/bin/env bash
# One-command paper-fair reproduction.
# Assumes:
#   - vLLM serving ChipMATE-V-9B on :8001, ChipMATE-P-9B on :8002 (see README)
#   - chipmate installed (pip install -e ..)
#   - iverilog + vvp on $PATH
#   - ChipBench dataset at $CHIPBENCH_DIR (default /opt/ChipBench/Verilog Gen)
#   - CVDP cid003 jsonl at $CVDP_JSONL
set -e
cd "$(dirname "$0")"

echo "================================================================"
echo "ChipMATE paper-fair reproduction — 4 benchmarks"
echo "================================================================"

echo; echo "[1/4] VerilogEval-v2 (~40 min)"
python3 vev2/spec_reword.py
python3 vev2/vev2_framework.py

echo; echo "[2/4] RTLLM-v2 (~13 min)"
python3 rtllm/rtllm_framework.py

echo; echo "[3/4] ChipBench self_contain + subsets (~25 min total)"
python3 chipbench/chipbench_framework.py
python3 chipbench/chipbench_framework_subsets.py
python3 chipbench/chipbench_score.py

echo; echo "[4/4] CVDP cid003 (~30 min)"
python3 cvdp/extract_cvdp_refsv.py
python3 cvdp/cvdp_framework.py --ref-sv-json /tmp/cvdp_refsv.json \
    --n-samples 5 --n-inner 5 --max-turns 3 \
    --out-name chipmate-9b-repro__cvdp_cid003

echo
echo "================================================================"
echo "Done. See each driver's stdout for the per-benchmark pass@1 / pass@5."
echo "================================================================"
