"""CVDP cid003 ChipMATE framework driver (V + P + cross_verify + multi-turn).

CVDP rows lack a `ref_sv` field — chipmate's run_problem needs one for port-list
extraction in cross_verify. We synthesize a stub ref_sv per problem by parsing
port directions from the cocotb harness Python (`dut.X.value = ...` writes
imply inputs; reads imply outputs; `int(dut.PARAM.value)` cast + runner's
`parameter = {...}` declarations imply parameters that get filtered out).

The synth ref_sv keeps the original toplevel module name so the V agent's
output (which follows the spec's module name) lines up with cocotb's
bench_row["toplevel"] without post-processing.

Pass@k = run_problem invoked once per (problem, seed); we run N_SAMPLES seeds.

Output: rollouts JSONL in eval_cvdp_verilog.py's expected format:
   {problem_id, samples: [{sample_idx, completion}, ...]}

Score: pipe through eval_cvdp_verilog.py and aggregate via pass@k estimator.
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sys.path.insert(0, '/ssd2/yichen/ChipMATE')

from chipmate.backends import OpenAICompatBackend
from chipmate.inference import run_problem


BENCH = '/home/yichen/workspace/LLaMA-Factory/data/cvdp_bench_verilog_VTRACK_cid003.jsonl'
OUT_DIR = '/ssd2/yichen/paper_eval/rollouts'
OUT_NAME = 'chipmate-9b-framework__cvdp_cid003'

V_URL = 'http://localhost:8001/v1'
P_URL = 'http://localhost:8002/v1'
V_MODEL = 'core12345/ChipMATE-V-9B'
P_MODEL = 'core12345/ChipMATE-P-9B'

INTERNAL = {'_log', '_name', '_path', '_id', '_handle'}
RESERVED_CLK = {'clk', 'clock', 'clk_in'}
RESERVED_RST = {'rst', 'rst_n', 'reset', 'reset_n', 'aresetn', 'arst', 'preset_n', 'presetn'}


def strip_py_comments(s):
    return '\n'.join(re.sub(r'(?<![\'"])#.*$', '', line) for line in s.split('\n'))


def _runner_params(harness):
    params = set()
    for k, v in harness.items():
        if 'runner' in k and k.endswith('.py'):
            for m in re.finditer(r'parameter\s*=\s*\{([^}]*)\}', v, re.DOTALL):
                for nm in re.finditer(r"['\"](\w+)['\"]\s*:", m.group(1)):
                    params.add(nm.group(1))
    return params


def extract_ports(harness):
    """Return (inputs, outputs) port-name lists. Best effort from cocotb test code."""
    test_code = ''
    for k, v in harness.items():
        if k.startswith('src/') and k.endswith('.py') and not k.endswith('runner.py'):
            test_code += '\n' + strip_py_comments(v)
    if not test_code:
        return [], []
    declared_params = _runner_params(harness)

    DA = r"dut(?:\.(\w+)|\[['\"](\w+)['\"]\])"
    writes = set()
    for m in re.finditer(DA + r"(?:\[[^\]]*\])*\.value\s*=(?!=)", test_code):
        writes.add(m.group(1) or m.group(2))
    for m in re.finditer(DA + r"(?:\[[^\]]*\])*\.setimmediatevalue", test_code):
        writes.add(m.group(1) or m.group(2))
    # Clock(dut.X, ...) → X is a clock input
    for m in re.finditer(r"Clock\s*\(\s*" + DA, test_code):
        writes.add(m.group(1) or m.group(2))
    # reset_dut(dut.X, ...) / reset(dut.X, ...) → X is reset input
    for m in re.finditer(r"reset[a-z_]*\s*\(\s*" + DA, test_code):
        writes.add(m.group(1) or m.group(2))
    # getattr(dut, "X") → assume input (we can't tell; counts as touched at least)
    for m in re.finditer(r"getattr\s*\(\s*dut\s*,\s*['\"](\w+)['\"]", test_code):
        writes.add(m.group(1))

    all_names = set()
    for m in re.finditer(DA, test_code):
        all_names.add(m.group(1) or m.group(2))
    all_names -= INTERNAL

    # Heuristic params: declared in runner OR (int() cast AND ALLCAPS_UNDERSCORE name)
    int_casts = set()
    for m in re.finditer(r"int\s*\(\s*" + DA + r"(?:\.value)?\s*\)", test_code):
        int_casts.add(m.group(1) or m.group(2))
    heuristic_caps = {n for n in int_casts if re.match(r'^[A-Z][A-Z0-9_]+$', n)}
    params = declared_params | heuristic_caps

    inputs = sorted(writes - params)
    outputs = sorted(all_names - writes - params)
    return inputs, outputs


def parse_spec_widths(question_text):
    """Best-effort: parse port widths from spec's Inputs/Outputs markdown tables.

    Looks for rows shaped `| name | width | description |` (or 4-col variant)
    under `### Inputs` / `### Outputs` headers. Returns {port_name: width_int}.
    Width is parsed as a plain int when present; expressions like `N*4` skip.
    """
    widths = {}
    sections = re.findall(
        r'###?\s*(?:Inputs?|Outputs?)\s*\n+((?:\|[^\n]*\n)+)',
        question_text, re.IGNORECASE,
    )
    for sec in sections:
        for row in sec.split('\n'):
            cells = [c.strip(' `*') for c in row.strip().strip('|').split('|')]
            if len(cells) < 2: continue
            name = cells[0]
            if not re.match(r'^\w+$', name): continue
            width_expr = cells[1]
            m = re.match(r'^(\d+)$', width_expr)
            if m:
                widths[name] = int(m.group(1))
                continue
            # [N-1:0] / N bits / N-bit forms
            m = re.search(r'\b(\d+)\s*[-]?\s*bit', width_expr)
            if m:
                widths[name] = int(m.group(1))
    return widths


def synth_ref_sv(toplevel, inputs, outputs, widths=None):
    """Build a stub `module <toplevel>(...);` with port directions + widths.

    `widths` (optional, from parse_spec_widths) overrides the default 32-bit
    width for ports it covers. Stub is consumed by chipmate.cross_verify's
    parse_ports / random-stimulus generator — narrower widths means more
    realistic stimuli and a higher chance cross_verify finds a discriminating
    input pattern.
    """
    if not inputs and not outputs:
        return None
    widths = widths or {}
    def fmt(direction, n):
        if n in RESERVED_CLK or n in RESERVED_RST:
            return f'  {direction} {n}'
        w = widths.get(n, 32)
        if w <= 1:
            return f'  {direction} {n}'
        return f'  {direction} [{w-1}:0] {n}'
    decls = [fmt('input', n) for n in inputs] + [fmt('output', n) for n in outputs]
    body = ',\n'.join(decls)
    return f'module {toplevel}(\n{body}\n);\nendmodule\n'


def get_user_question(row):
    for m in row['question']:
        if m.get('role') == 'user':
            return m['content']
    return ''


def wrap_for_cocotb(verilog_raw, expected_toplevel):
    """Wrap chipmate's bare-verilog output back into a code-fenced completion.

    Also rewrite the first `module <name>` to match `expected_toplevel` so
    cocotb (which loads HDL_TOPLEVEL from bench_row["toplevel"]) finds it.
    """
    if not verilog_raw or 'module' not in verilog_raw:
        return ''
    m = re.search(r'\bmodule\s+(\w+)', verilog_raw)
    if m and m.group(1) != expected_toplevel:
        old = m.group(1)
        verilog_raw = re.sub(r'\bmodule\s+' + re.escape(old),
                             f'module {expected_toplevel}', verilog_raw, count=1)
        verilog_raw = re.sub(r'endmodule\s*:\s*' + re.escape(old) + r'\b',
                             f'endmodule : {expected_toplevel}', verilog_raw)
    return f'```verilog\n{verilog_raw}\n```'


def run_one_sample(task_id, question, ref_sv, v_backend, p_backend, n_inner, max_turns, seed, num_verify_tests):
    try:
        r = run_problem(
            task_id=task_id, question=question, ref_sv=ref_sv,
            v_backend=v_backend, p_backend=p_backend,
            n=n_inner, max_turns=max_turns,
            num_verify_tests=num_verify_tests,
            seed=seed,
        )
        return r.verilog
    except Exception as e:
        sys.stderr.write(f'[seed {seed}] {task_id}: {e}\n')
        return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-samples', type=int, default=5, help='pass@k samples (framework runs per problem)')
    ap.add_argument('--n-inner', type=int, default=3, help='candidates per turn inside framework')
    ap.add_argument('--max-turns', type=int, default=2)
    ap.add_argument('--num-verify-tests', type=int, default=30)
    ap.add_argument('--workers', type=int, default=4, help='parallel problems × samples')
    ap.add_argument('--limit', type=int, default=0, help='process first N problems (0=all)')
    ap.add_argument('--only', default='', help='comma list of problem_ids; empty=all')
    ap.add_argument('--out-name', default=OUT_NAME)
    ap.add_argument('--ref-sv-json', default='',
                    help='path to {problem_id: ref_sv_text} JSON; if set, used in place of synth_ref_sv (proper port widths from prior candidates)')
    args = ap.parse_args()
    refsv_override = {}
    if args.ref_sv_json:
        refsv_override = json.load(open(args.ref_sv_json))
        sys.stderr.write(f'  ref_sv_json: loaded {len(refsv_override)} per-problem stubs\n')

    os.makedirs(OUT_DIR, exist_ok=True)
    raw_out = os.path.join(OUT_DIR, args.out_name + '.raw.jsonl')

    rows = [json.loads(l) for l in open(BENCH)]
    if args.only:
        keep = set(args.only.split(','))
        rows = [r for r in rows if r['problem_id'] in keep]
    if args.limit:
        rows = rows[:args.limit]
    print(f'CVDP framework: {len(rows)} problems × {args.n_samples} samples', file=sys.stderr)

    v_backend = OpenAICompatBackend(model=V_MODEL, api_key='EMPTY', base_url=V_URL)
    p_backend = OpenAICompatBackend(model=P_MODEL, api_key='EMPTY', base_url=P_URL)

    # Resume support: skip (task, sample) already in raw_out
    done = set()
    if os.path.exists(raw_out):
        for l in open(raw_out):
            try:
                r = json.loads(l)
                done.add((r['task_id'], r['sample_idx']))
            except Exception:
                pass
    if done:
        print(f'  resume: {len(done)} (task, sample) already done', file=sys.stderr)

    jobs = []
    n_no_ports = 0
    toplevel_by_tid = {}
    for r in rows:
        toplevel_by_tid[r['problem_id']] = r['toplevel']
        # Prefer per-problem override (proper port widths from prior candidates)
        if r['problem_id'] in refsv_override:
            ref_sv = refsv_override[r['problem_id']]
        else:
            ins, outs = extract_ports(r['harness'])
            q_text = get_user_question(r)
            widths = parse_spec_widths(q_text)
            ref_sv = synth_ref_sv(r['toplevel'], ins, outs, widths)
        if ref_sv is None:
            n_no_ports += 1
            sys.stderr.write(f'  NO PORTS: {r["problem_id"]} — skipping\n')
            continue
        question = get_user_question(r)
        for s in range(args.n_samples):
            if (r['problem_id'], s) in done:
                continue
            jobs.append((r['problem_id'], s, question, ref_sv))
    print(f'  {len(jobs)} jobs todo  ({n_no_ports} problems skipped — no ports)', file=sys.stderr)

    f_raw = open(raw_out, 'a')
    t0 = time.time()
    n_completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                run_one_sample, tid, q, rsv, v_backend, p_backend,
                args.n_inner, args.max_turns, s, args.num_verify_tests,
            ): (tid, s) for (tid, s, q, rsv) in jobs
        }
        for fut in as_completed(futs):
            tid, s = futs[fut]
            try:
                verilog = fut.result() or ''
            except Exception as e:
                verilog = ''
                sys.stderr.write(f'[{tid} seed={s}] worker error: {e}\n')
            completion = wrap_for_cocotb(verilog, toplevel_by_tid[tid])
            f_raw.write(json.dumps({
                'task_id': tid, 'sample_idx': s, 'completion': completion,
            }) + '\n')
            f_raw.flush()
            n_completed += 1
            if n_completed % 10 == 0:
                el = time.time() - t0
                rate = n_completed / el if el else 0
                eta = (len(jobs) - n_completed) / rate if rate else 0
                sys.stderr.write(f'  [{n_completed}/{len(jobs)}] rate={rate:.2f}/s eta={eta:.0f}s\n')
    f_raw.close()

    # Aggregate raw into per-problem samples list for eval_cvdp_verilog.py
    agg_out = os.path.join(OUT_DIR, args.out_name + '.jsonl')
    groups = defaultdict(dict)
    for l in open(raw_out):
        r = json.loads(l)
        groups[r['task_id']][r['sample_idx']] = r['completion']
    with open(agg_out, 'w') as f:
        for tid, by_idx in groups.items():
            samples = [{'sample_idx': i, 'completion': by_idx[i]} for i in sorted(by_idx)]
            f.write(json.dumps({'problem_id': tid, 'samples': samples}) + '\n')
    print(f'\nDONE — {agg_out}', file=sys.stderr)
    print(f'Next: python3 /home/yichen/workspace/LLaMA-Factory/scripts/eval_cvdp_verilog.py \\\n'
          f'         --bench {BENCH} \\\n'
          f'         --rollouts {agg_out} \\\n'
          f'         --out {agg_out.replace(".jsonl", ".scored.jsonl")} \\\n'
          f'         --parallel 16 --timeout 60', file=sys.stderr)


if __name__ == '__main__':
    main()
