# ChipMATE

Multi-agent inference framework for RTL code generation with LLM cross-verification.

- **Website:** [https://chipmate.picasso-lab.com](https://chipmate.picasso-lab.com)
- **Models:** [core12345/ChipMATE-V-4B](https://huggingface.co/core12345/ChipMATE-V-4B) · [core12345/ChipMATE-P-4B](https://huggingface.co/core12345/ChipMATE-P-4B) · [core12345/ChipMATE-V-9B](https://huggingface.co/core12345/ChipMATE-V-9B) · [core12345/ChipMATE-P-9B](https://huggingface.co/core12345/ChipMATE-P-9B)

ChipMATE pairs a **Verilog-generating agent** with a **Python reference-model agent** that mutually verify each other's outputs on random stimuli. The two agents iteratively refine their code through cross-verification feedback until they agree. No golden testbench, no human spec annotations, no API dependency at deployment time.

**No golden testbench.** Many prior RTL generation workflows rely on a pre-written golden testbench for iterative self-correction, which is rarely available in real chip design workflows. ChipMATE never uses golden testbenches when generating codes, making the framework directly applicable to industry settings where only a natural-language specification exists.

**Interface I/O signals.** In real chip design, port names and bit-widths are formally specified before RTL implementation begins. To reflect this, ChipMATE conditions its prompts on a structured interface specification. Since benchmarks do not provide one, we extract the module port declaration from the golden Verilog file. No implementation logic is ever read to guarantee fairness.

## Benchmark results

Pass@1 / pass@5 (%) for Verilog generation, taken from Table 1 of our paper:

| Model | Size | VerilogEval V2 | RTLLM V2 | ChipBench-SC | CVDP cid03 |
|---|---:|---:|---:|---:|---:|
| GPT-4o | – | 64.1 / 73.7 | 56.5 / 70.3 | 20.0 / 33.3 | 39.0 / 40.4 |
| GPT-5.5 | – | 84.7 / 90.4 | 63.2 / 68.0 | 30.7 / 36.7 | 44.0 / 48.7 |
| Claude Opus 4.7 | – | 86.9 / 90.4 | 64.8 / 68.0 | 31.3 / 46.7 | 42.8 / 47.9 |
| DeepSeek Coder | 236B | 68.5 / 80.8 | 57.6 / 70.0 | 16.7 / 30.0 | 22.3 / 37.2 |
| DeepSeek V4 | 1.6T | 67.3 / 80.1 | 58.8 / 66.0 | 18.0 / 36.7 | 21.5 / 34.6 |
| DeepSeek R1 | 671B | 77.5 / 84.7 | 64.7 / 75.8 | 26.7 / 40.0 | 27.7 / 42.1 |
| CodeV-R1 (distill) | 7B | 65.2 / 75.2 | 57.2 / 71.9 | 13.3 / 26.7 | 26.2 / 42.1 |
| CodeV-R1 | 7B | 68.8 / 78.2 | 68.0 / **78.2** | 30.0 / 40.0 | 26.8 / 43.3 |
| Qwen3.5-4B (base) | 4B | 41.7 / 60.9 | 34.3 / 49.7 | 6.7 / 10.0 | 11.8 / 13.9 |
| Qwen3.5-9B (base) | 9B | 48.5 / 66.6 | 36.1 / 57.8 | 13.3 / 20.0 | 13.3 / 21.5 |
| **ChipMATE-Agents-4B** | 4B | 75.0 / 76.3 | 74.6 / 77.3 | 33.3 / 43.3 | 32.1 / 41.3 |
| **ChipMATE-Agents-9B** | 9B | 80.1 / 82.4 | 75.8 / 77.3 | 36.7 / 43.3 | 40.4 / 44.6 |

`ChipMATE-Agents-X` is the full multi-agent system you get when you run this repo — Verilog and Python agents cooperating via cross-verification.

## Paper protocol (Table 1)

One configuration produces every agentic row in Table 1. To reproduce a Table 1 number, use exactly these settings — they are what the commands in this README pass:

| Setting | Value | CLI flag |
|---|---|---|
| Candidate (V, P) pairs per turn | **N = 3** (Best-of-3) | `-n 3` |
| Max cross-verification turns | **T = 3** | `-t 3` |
| Random stimuli per cross-verify call | **1000** | `--num-verify-tests 1000` |
| Sampling temperature | **1.0** | `--temperature 1.0` |
| Responses per problem (pass@k scoring) | 10 independent workflow runs, unbiased pass@k estimator | run the CLI 10× with distinct `--seed` |

The API baseline rows in Table 1 (GPT/Claude/DeepSeek/CodeV-R1) are **single-pass generations** — one sampled response per query per run under the same 10-run pass@k protocol — not the multi-agent workflow.

## Install

```bash
git clone git@github.com:zhongkaiyu/ChipMATE.git
cd ChipMATE
pip install -e .
# Optional: enable the Claude/Anthropic backend
pip install -e ".[anthropic]"
```

You also need `iverilog` (with `vvp`) on `PATH` — the cross-verification harness compiles Verilog DUTs with Icarus Verilog. On Debian/Ubuntu:

```bash
sudo apt-get install iverilog
```

Or skip the host install and use the [Docker image](#docker), which ships iverilog out of the box.

## Quickstart

There are two ways to drive ChipMATE. **Option A** is the path described in the paper and produces the headline numbers above. **Option B** is a convenience path for using a hosted LLM as both agents.

---

### Option A. Use our open-source ChipMATE weights on your own GPUs *(reproduces the paper)*

Serve the Verilog agent and the Python reference-model agent as two local OpenAI-compatible endpoints with [vLLM](https://github.com/vllm-project/vllm), then drive them with the ChipMATE loop.

**Step 1 — serve the two agents.** Pick the size you want (4B or 9B). Two GPUs are enough for the 9B pair:

```bash
# Terminal 1 — Verilog agent on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve core12345/ChipMATE-V-9B --port 8001

# Terminal 2 — Python reference-model agent on GPU 1
CUDA_VISIBLE_DEVICES=1 vllm serve core12345/ChipMATE-P-9B --port 8002
```

For the 4B pair, replace `9B` with `4B`. Both fit on a single 24GB consumer GPU.

**Step 2a — run one problem from Python.**

```python
from chipmate import make_backend, run_problem

v_backend = make_backend(model="core12345/ChipMATE-V-9B",
                         base_url="http://localhost:8001/v1", api_key="dummy")
p_backend = make_backend(model="core12345/ChipMATE-P-9B",
                         base_url="http://localhost:8002/v1", api_key="dummy")

result = run_problem(
    task_id="my_problem",
    question="Implement a Verilog module that ...",
    ref_sv="module top_module(input clk, ...);\nendmodule\n",
    v_backend=v_backend,
    p_backend=p_backend,
    # Paper protocol (Table 1):
    n=3, max_turns=3, num_verify_tests=1000, temperature=1.0,
)
print(result.verilog)             # final Verilog implementation
print(result.matched)             # True iff cross-verify reached match_rate == 1.0
```

**Step 2b — reproduce the paper's VerilogEval V2 pass@1 from the command line.** The paper's reported configuration is Best-of-3 over 3 turns with 1000 stimuli per cross-check (see [Paper protocol](#paper-protocol-table-1)):

```bash
chipmate \
  --input        examples/verilogeval_v2.jsonl \
  --out          results/chipmate-agents-9b__verilogeval_v2.jsonl \
  --provider     openai-compat \
  --model        core12345/ChipMATE-V-9B \
  --base-url     http://localhost:8001/v1 \
  --api-key      dummy \
  --p-model      core12345/ChipMATE-P-9B \
  --p-base-url   http://localhost:8002/v1 \
  --p-api-key    dummy \
  -n 3 -t 3 --num-verify-tests 1000 --temperature 1.0 \
  --workers 4
```

The CLI is just a batch driver around `run_problem`. The input JSONL has one `{task_id, question, ref_sv}` per line — `ref_sv` is reference Verilog whose port list defines the interface (its body is never compared against, only the port declarations are read). [`examples/verilogeval_v2.jsonl`](examples/verilogeval_v2.jsonl) ships the full 156-problem VerilogEval V2 set in this format. Pipe the resulting JSONL through your favorite Verilog testbench to get pass@k.

---

### Option B. Drive ChipMATE with a hosted LLM API (OpenAI / DeepSeek / Claude / Gemini)

If you don't want to host the weights, you can use any frontier API as both agents. Note: the API-based rows in the benchmark table above are **single-pass baselines** (one sampled response per query, per the paper's protocol) — this Option B path is a convenience for running the ChipMATE workflow with hosted models, not how those baseline rows were produced.

**Step 1 — pick a provider and run one problem.**

```python
from chipmate import make_backend, run_problem

# DeepSeek — speaks the OpenAI Chat-Completions API.
backend = make_backend(
    provider="openai-compat",
    model="deepseek-chat",
    api_key="sk-...",                         # or set OPENAI_API_KEY
    base_url="https://api.deepseek.com",
)

result = run_problem(
    task_id="my_problem",
    question="Implement a Verilog module that ...",
    ref_sv="module top_module(input clk, ...);\nendmodule\n",
    v_backend=backend,    # same backend drives both agents
)
print(result.verilog)
```

Other providers share the same interface — just change `provider` / `model` / `base_url`:

| Provider           | `provider`         | `base_url`                                                       |
|--------------------|--------------------|------------------------------------------------------------------|
| OpenAI / GPT       | `openai-compat`    | `https://api.openai.com/v1`                                      |
| DeepSeek           | `openai-compat`    | `https://api.deepseek.com`                                       |
| Google Gemini      | `openai-compat`    | `https://generativelanguage.googleapis.com/v1beta/openai/`       |
| Anthropic / Claude | `anthropic`        | *(native; install `chipmate-inference[anthropic]`)*              |

**Step 2 — batch via the CLI.** Same `chipmate` command as in Option A, just point it at the hosted endpoint:

```bash
chipmate \
  --input    examples/verilogeval_v2.jsonl \
  --out      results/deepseek__verilogeval_v2.jsonl \
  --provider openai-compat \
  --model    deepseek-chat \
  --base-url https://api.deepseek.com \
  --api-key  $DEEPSEEK_API_KEY \
  -n 3 -t 3 --num-verify-tests 1000 \
  --workers 4
```

---

## Defaults vs. paper protocol

The package defaults (`n=10`, `max_turns=5`, `num_verify_tests=30`) are a **fast exploratory setting** for interactive use — they are *not* the paper's configuration. Every Table 1 agentic number uses the paper protocol above: `-n 3 -t 3 --num-verify-tests 1000 --temperature 1.0` (Best-of-3, T=3, selected in the paper's inference-parameter sweep). Always pass these flags explicitly when reproducing or comparing against the paper.

## Docker

```bash
docker build -t chipmate .

docker run --rm \
  -e DEEPSEEK_API_KEY=sk-... \
  -v "$PWD":/work \
  chipmate \
    --input    /work/problems.jsonl \
    --out      /work/results.jsonl \
    --provider openai-compat \
    --model    deepseek-chat \
    --base-url https://api.deepseek.com \
    -n 3 -t 3 --num-verify-tests 1000
```

The image ships `iverilog` so you don't need to install it on the host.

## Contact

If you have any questions or would like further information, please feel free to contact us at **zhy055@ucsd.edu** and **yil384@ucsd.edu**. You can also visit our homepages for more details about our work: [Zhongkai Yu](https://zhongkaiyu.github.io/) and [Yichen Lin](https://yil384.github.io/).

## License

Apache 2.0. See [LICENSE](LICENSE).
