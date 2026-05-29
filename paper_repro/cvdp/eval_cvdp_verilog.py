#!/usr/bin/env python3
"""CVDP Verilog-track eval runner.

For each model sample, place the extracted Verilog in the DUT path expected
by CVDP's testbench, run cocotb+iverilog, and collect pass/fail.

Uses cocotb 2.x (from /home/yichen/.local). Some CVDP tbs are written in
cocotb 1.x style (dut.sig[idx] handle indexing / .signed_integer) — those
will show up as FAIL with a traceback in the cocotb log. Treat as
"tb_incompat" and report alongside real functional failures.

Input:
    --bench    : data/cvdp_bench_verilog.jsonl (harness + metadata)
    --rollouts : jsonl of {problem_id, samples:[{sample_idx, completion}]}
    --out      : scored jsonl

Usage:
    python scripts/eval_cvdp_verilog.py \\
        --bench data/cvdp_bench_verilog.jsonl \\
        --rollouts rollout.jsonl \\
        --out scored.jsonl \\
        --parallel 16 --timeout 180
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


RUNNER_TEMPLATE = r'''
import os, sys
try:
    from cocotb_tools.runner import get_runner
except ImportError:
    from cocotb.runner import get_runner

sim = {sim!r}
toplevel = {toplevel!r}
module = {module!r}
sources = {sources!r}
work_dir = {work_dir!r}

sys.path.insert(0, os.path.join(work_dir, "src"))

# Some CVDP testbenches read runtime inputs via cocotb.plusargs["NAME"].
# The benchmark's own test_runner.py supplies these as random +NAME=value
# plusargs; our generic runner must do the same or the test raises KeyError.
import re as _re, random as _random
_plusargs = []
_test_path = os.path.join(work_dir, "src", module + ".py")
if os.path.exists(_test_path):
    _src = open(_test_path).read()
    _names = set(_re.findall(r'plusargs\s*\[\s*[\'"](\w+)[\'"]\s*\]', _src))
    for _nm in sorted(_names):
        _plusargs.append("+{{}}={{}}".format(_nm, _random.randint(0, 255)))

runner = get_runner(sim)
runner.build(
    sources=sources,
    hdl_toplevel=toplevel,
    always=True,
    clean=True,
    waves=False,
    timescale=("1ns", "1ps"),
    build_dir=os.path.join(work_dir, "sim_build"),
)
runner.test(
    hdl_toplevel=toplevel,
    test_module=module,
    test_dir=os.path.join(work_dir, "results"),
    plusargs=_plusargs,
)
'''


def extract_verilog(text):
    for tag in ("verilog", "systemverilog", "sv"):
        ms = list(re.finditer(rf"```{tag}\s*([\s\S]*?)\s*```", text, re.IGNORECASE))
        if ms:
            return ms[-1].group(1).strip()
    m = list(re.finditer(r"```\s*([\s\S]*?)\s*```", text))
    return m[-1].group(1).strip() if m else text.strip()


def remap_source_path(code_path, work_dir):
    """Map CVDP's /code/... path to our local work_dir."""
    if code_path.startswith("/code/"):
        return os.path.join(work_dir, code_path[len("/code/"):])
    return os.path.join(work_dir, code_path.lstrip("/"))


def parse_results_xml(xml_path):
    """cocotb writes results.xml as JUnit-ish. Return (n_pass, n_fail, n_err, summary)."""
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        return 0, 0, 0, f"xml_parse_error: {e}"
    root = tree.getroot()
    n_pass = n_fail = n_err = 0
    msgs = []
    # cocotb 2.x: <testsuites><testsuite><testcase ...>  with <failure/>/<error/>
    for tc in root.iter("testcase"):
        fail = tc.find("failure")
        err = tc.find("error")
        if fail is not None:
            n_fail += 1
            msgs.append(f"FAIL {tc.get('name')}: {(fail.get('message') or '')[:120]}")
        elif err is not None:
            n_err += 1
            msgs.append(f"ERR  {tc.get('name')}: {(err.get('message') or '')[:120]}")
        else:
            n_pass += 1
    return n_pass, n_fail, n_err, " | ".join(msgs)[:400]


def run_one(python_bin, bench_row, completion, timeout):
    """Run a single (problem, completion) in an isolated tempdir.
    Returns (passed: bool, err: str|None, detail: dict)."""
    dut_sv = extract_verilog(completion)
    if not dut_sv or "module" not in dut_sv.lower():
        return False, "no verilog module extracted", {}

    with tempfile.TemporaryDirectory(prefix="cvdp_") as work_dir:
        # 1) drop harness files under work_dir/src/
        for k, v in bench_row["harness"].items():
            path = os.path.join(work_dir, k)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(v)

        # 2) write DUT verilog to path(s) the harness expects
        local_sources = []
        for s in bench_row["verilog_sources"]:
            p = remap_source_path(s, work_dir)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(dut_sv)
            local_sources.append(p)

        # 3) write our runner
        runner_script = RUNNER_TEMPLATE.format(
            sim=bench_row["sim"], toplevel=bench_row["toplevel"],
            module=bench_row["module"], sources=local_sources,
            work_dir=work_dir,
        )
        runner_path = os.path.join(work_dir, "run.py")
        with open(runner_path, "w") as f:
            f.write(runner_script)

        # 4) exec
        try:
            r = subprocess.run(
                [python_bin, runner_path],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout >{timeout}s", {}

        xml = os.path.join(work_dir, "results", "results.xml")
        if not os.path.exists(xml):
            stderr_tail = (r.stderr or "")[-600:]
            return False, f"no results.xml (exit={r.returncode}); {stderr_tail}", {
                "stderr_tail": stderr_tail
            }

        n_pass, n_fail, n_err, summary = parse_results_xml(xml)
        passed = (n_fail == 0 and n_err == 0 and n_pass > 0)
        detail = {"n_pass": n_pass, "n_fail": n_fail, "n_err": n_err,
                  "summary": summary}
        return passed, None if passed else summary[:200], detail


def _worker(args):
    python_bin, bench_row, pid, sample_idx, completion, timeout = args
    try:
        passed, err, detail = run_one(python_bin, bench_row, completion, timeout)
    except Exception as e:
        passed, err, detail = False, f"worker_exc: {e}"[:200], {}
    return pid, sample_idx, passed, err, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--python-bin", default="/home/yichen/.local/bin/python",
                    help="python with cocotb installed")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=180,
                    help="per-sample cocotb run timeout (s)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only score first N problems (0=all)")
    args = ap.parse_args()

    bench_by_pid = {}
    with open(args.bench) as f:
        for line in f:
            r = json.loads(line)
            bench_by_pid[r["problem_id"]] = r

    rollouts = []
    with open(args.rollouts) as f:
        for line in f:
            rollouts.append(json.loads(line))
    if args.limit > 0:
        rollouts = rollouts[:args.limit]

    jobs = []
    for r in rollouts:
        pid = r["problem_id"]
        if pid not in bench_by_pid:
            continue
        for s in r["samples"]:
            jobs.append((args.python_bin, bench_by_pid[pid], pid,
                         s["sample_idx"], s["completion"], args.timeout))
    print(f"jobs: {len(jobs)} across {len(rollouts)} problems", file=sys.stderr)

    verdicts = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futs = [pool.submit(_worker, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            pid, sidx, passed, err, detail = fut.result()
            verdicts[pid][sidx] = (passed, err, detail)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                npass = sum(1 for v in verdicts.values()
                            for p, _, _ in v.values() if p)
                print(f"  {done}/{len(jobs)}  passing={npass}", file=sys.stderr)

    n_all_pass = n_unstable = n_dead = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rollouts:
            pid = r["problem_id"]
            for s in r["samples"]:
                v = verdicts[pid].get(s["sample_idx"], (False, "no verdict", {}))
                s["passed"], s["err"], s["detail"] = v
            n_pass = sum(1 for s in r["samples"] if s["passed"])
            n = len(r["samples"])
            r["n_pass"], r["n_total"] = n_pass, n
            r["pass_rate"] = n_pass / n if n else 0.0
            if n_pass == n:
                n_all_pass += 1
            elif n_pass == 0:
                n_dead += 1
            else:
                n_unstable += 1
            f.write(json.dumps(r) + "\n")

    n = len(rollouts)
    print("\n" + "=" * 60)
    print(f"Problems: {n}")
    if n:
        print(f"  all pass      : {n_all_pass} ({n_all_pass/n*100:.1f}%)")
        print(f"  unstable 0<p<1: {n_unstable} ({n_unstable/n*100:.1f}%)")
        print(f"  all fail      : {n_dead} ({n_dead/n*100:.1f}%)")
    print(f"→ {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
