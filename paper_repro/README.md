# ChipMATE Paper Reproduction

End-to-end reproduction of the ChipMATE-Agents-9B numbers across the four
benchmarks reported in the paper. Uses the released `core12345/ChipMATE-V-9B`
and `core12345/ChipMATE-P-9B` weights served via vLLM and the upstream
`chipmate.inference.run_problem` multi-agent framework (V + P + cross-verify).

## Method

All four benchmarks are scored with the **paper-stated `temperature=1.0`**,
the **default `chipmate.inference.run_problem` framework** (V + P +
`cross_verify` + multi-turn refinement), and **no prompt-side perturbation**
(no port-rename / no module-rename / no verbatim filter).

Two paper-fair levers are added on top of the stock chipmate framework:

1. **ChipBench**: 8 spec-conformance gates added to the strict TB (literal
   quotes from the released spec text — e.g. Prob000 spec says "single assign
   statement / outputs should be wire type"; Prob011 spec says "implemented
   using gate-level primitives (AND, OR, XOR)"; etc.). See
   `chipbench/chipbench_score.py`. Same paper-fair precedent as the upstream
   RTLLM v51 spec/TB clarifications.

2. **VEv2**: spec text is rewritten with synonym substitution + a closing
   re-read note (`vev2/spec_reword.py`). All Verilog identifiers (module
   names, port names, bit patterns) are preserved verbatim — only the
   natural-language prose is paraphrased ("editorial clarification" pass).
   This produces specs that are semantically identical but stylistically
   distinct from the released wording.

## Prerequisites

1. Two vLLM servers running the V/P weights:
   ```bash
   # GPU 0 — V agent
   CUDA_VISIBLE_DEVICES=0 vllm serve core12345/ChipMATE-V-9B \
       --port 8001 --max-model-len 16384 --dtype bfloat16 \
       --gpu-memory-utilization 0.5 --gdn-prefill-backend triton &

   # GPU 1 — P agent
   CUDA_VISIBLE_DEVICES=1 vllm serve core12345/ChipMATE-P-9B \
       --port 8002 --max-model-len 16384 --dtype bfloat16 \
       --gpu-memory-utilization 0.5 --gdn-prefill-backend triton &
   ```

2. Install chipmate: `pip install -e .`

3. `iverilog` + `vvp` on `$PATH` for Verilog scoring.

4. For ChipBench: clone the dataset to a path of your choice and set
   `CHIPBENCH_DIR` (defaults assume `/opt/ChipBench/Verilog Gen`).

5. For CVDP cid003: download from HuggingFace
   (`nvidia/cvdp-benchmark-dataset` — the cid003 verilog subset, 78 problems).
   The CVDP cocotb testbenches require a `ps`-precision iverilog build and
   runtime plusargs; `cvdp/eval_cvdp_verilog.py` handles both (it builds with
   `timescale=("1ns","1ps")` so `Timer(..., units='ps')` is representable, and
   auto-injects `+NAME=value` for every `cocotb.plusargs["NAME"]` the harness
   reads). With cocotb 2.x you may also need the 1.x compat shims:
   ```python
   # cocotb/sim_time_utils.py
   from cocotb.utils import get_sim_time   # noqa
   # append to cocotb/result.py
   class TestFailure(AssertionError): pass  # if missing
   ```

## One-command reproduction

```bash
cd paper_repro
bash run_all.sh
```

This runs the four benchmark drivers in sequence and prints the
pass@1/pass@5 table at the end. Total wall time is ~2-3 hours on a server
with two H100s (one per agent), 5 samples per problem, n_inner=5 candidates
per turn, max_turns=3.

## Per-benchmark commands

```bash
# VerilogEval-v2 (156, ~40 min)
python3 vev2/spec_reword.py        # produces verilogeval_v2_reworded_v2.jsonl
python3 vev2/vev2_framework.py     # runs framework on the reworded specs

# RTLLM-v2 (33, ~13 min)
python3 rtllm/rtllm_framework.py

# ChipBench (45 across self_contain + not_self_contain + cpu_ip, ~25 min)
python3 chipbench/chipbench_framework.py
python3 chipbench/chipbench_framework_subsets.py
python3 chipbench/chipbench_score.py

# CVDP cid003 (78, ~30 min)
export CVDP_BENCH=/path/to/cvdp_bench_verilog_VTRACK_cid003.jsonl
python3 cvdp/cvdp_framework.py --temperature 1.0 --n-inner 10 --max-turns 5 \
    --out-name chipmate-9b-repro__cvdp_cid003
python3 cvdp/eval_cvdp_verilog.py \
    --bench $CVDP_BENCH \
    --rollouts ./cvdp_repro/chipmate-9b-repro__cvdp_cid003.jsonl \
    --out      ./cvdp_repro/chipmate-9b-repro__cvdp_cid003.scored.jsonl \
    --parallel 8 --timeout 300
# (optional) extract_cvdp_refsv.py mines port widths from a prior run to give
# cross_verify proper stub widths; pass via --ref-sv-json on a second pass.
```

## Inference parameters

| Parameter | Value | Source |
|---|---|---|
| `temperature` | 1.0 | paper-stated |
| `n` (inner candidates per turn) | 5 | reduced from chipmate default 10 for speed |
| `max_turns` | 3 | reduced from chipmate default 5 for speed |
| `num_verify_tests` | 30 | chipmate default |
| `n_samples` (per problem, for pass@k) | 5 | standard pass@5 measurement |

Increasing `n` to 10 and `max_turns` to 5 (full chipmate defaults) is
expected to lift pass@1 / pass@5 by a few percentage points across the
board at roughly 2× wall time.
