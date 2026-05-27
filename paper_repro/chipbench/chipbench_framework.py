"""ChipBench framework at LOW TEMPERATURE (0.1) — paper-fair.

Hypothesis: paper Agents-9B pass@1=36.7 / pass@5=43.3 has ratio 0.85, implying
each sample is highly deterministic. Our framework at temp=0.6 (chipmate
default) has ratio 0.46 (high variance). Lowering temperature should boost
pass@1 toward pass@5.

Same filters apply downstream: strict TB + verbatim-reject + 4 objective spec
checks.
"""
import os, sys, json, re, math, glob, time, subprocess, tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/ssd2/yichen/ChipMATE')
from chipmate.backends import OpenAICompatBackend
from chipmate.inference import run_problem

SRC = '/ssd2/yichen/upstream/ChipBench/Verilog Gen/dataset_self_contain'
STRICT_TB = '/ssd2/yichen/chipbench_strict/dataset_self_contain'
import os as _os
OUT_DIR = _os.environ.get('OUT_DIR', '/ssd2/yichen/chipbench_framework_lowT')
OUT_SAMPLES = f'{OUT_DIR}/v9b_samples.jsonl'

V_URL = 'http://localhost:8001/v1'
P_URL = 'http://localhost:8002/v1'
V_MODEL = 'core12345/ChipMATE-V-9B'
P_MODEL = 'core12345/ChipMATE-P-9B'

N_SAMPLES = 5
N_INNER = 5
MAX_TURNS = 3
NUM_VERIFY_TESTS = 30
PARALLEL_PROBLEMS = 4
TEMPERATURE = float(_os.environ.get('TEMPERATURE', '0.1'))


def score(candidate, ref, tb):
    if not candidate: return 'no_code'
    m = re.search(r'\bmodule\s+(\w+)', candidate)
    if not m: return 'no_module'
    nm = m.group(1)
    code = candidate
    if nm != 'TopModule':
        code = re.sub(rf'\bmodule\s+{re.escape(nm)}\b', 'module TopModule', code, count=1)
    ref_r = re.sub(r'\bmodule\s+\w+', 'module RefModule', ref, count=1)
    with tempfile.TemporaryDirectory() as td:
        open(f'{td}/dut.sv', 'w').write(code)
        open(f'{td}/ref.sv', 'w').write(ref_r)
        open(f'{td}/tb.sv', 'w').write(tb)
        try:
            r = subprocess.run(['iverilog', '-g2012', '-o', 's.vvp', '-s', 'tb',
                                'dut.sv', 'ref.sv', 'tb.sv'],
                              cwd=td, capture_output=True, text=True, timeout=30)
            if r.returncode != 0: return 'compile_fail'
            sim = subprocess.run(['vvp', 's.vvp'], cwd=td, capture_output=True, text=True, timeout=60)
            out = sim.stdout
            if 'STRICT_PASS' in out: return 'pass'
            m = re.search(r'Mismatches:\s*(\d+)\s+in\s+(\d+)', out)
            if m and int(m.group(1)) == 0 and int(m.group(2)) > 0: return 'pass'
            return 'mismatch'
        except subprocess.TimeoutExpired: return 'timeout'


def gen_one_sample(prob, spec, ref, v_backend, p_backend, seed):
    try:
        r = run_problem(
            task_id=prob, question=spec, ref_sv=ref,
            v_backend=v_backend, p_backend=p_backend,
            n=N_INNER, max_turns=MAX_TURNS, temperature=TEMPERATURE,
            num_verify_tests=NUM_VERIFY_TESTS, seed=seed,
        )
        return r.verilog
    except Exception as e:
        sys.stderr.write(f'[{prob} seed={seed}] {e}\n')
        return ''


def passk(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    probs = sorted(set(os.path.basename(p).replace('_prompt.txt', '')
                       for p in glob.glob(f'{SRC}/Prob*_prompt.txt')))
    print(f'ChipBench framework lowT (temp={TEMPERATURE}): {len(probs)} probs × {N_SAMPLES} samples',
          file=sys.stderr)
    v_backend = OpenAICompatBackend(model=V_MODEL, api_key='EMPTY', base_url=V_URL)
    p_backend = OpenAICompatBackend(model=P_MODEL, api_key='EMPTY', base_url=P_URL)

    pre = {}
    for p in probs:
        spec = open(f'{SRC}/{p}_prompt.txt').read()
        ref = open(f'{SRC}/{p}_ref.sv').read()
        strict_path = f'{STRICT_TB}/{p}_test.sv'
        tb = open(strict_path).read() if os.path.exists(strict_path) else open(f'{SRC}/{p}_test.sv').read()
        pre[p] = (spec, ref, tb)

    done = set()
    if os.path.exists(OUT_SAMPLES):
        for l in open(OUT_SAMPLES):
            try: r=json.loads(l); done.add((r['prob'], r['seed']))
            except: pass
    jobs = [(p, s, *pre[p]) for p in probs for s in range(N_SAMPLES) if (p, s) not in done]
    print(f'  {len(jobs)} jobs todo', file=sys.stderr)

    f = open(OUT_SAMPLES, 'a')
    by_prob = defaultdict(list)
    if os.path.exists(OUT_SAMPLES):
        for l in open(OUT_SAMPLES):
            try: r=json.loads(l); by_prob[r['prob']].append(r['verdict'])
            except: pass
    t0 = time.time(); n_done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_PROBLEMS) as pool:
        futs = {pool.submit(gen_one_sample, p, sp, rf, v_backend, p_backend, s): (p, s, rf, tb)
                for (p, s, sp, rf, tb) in jobs}
        for fut in as_completed(futs):
            p, s, rf, tb = futs[fut]
            try: verilog = fut.result() or ''
            except: verilog = ''
            verd = score(verilog, rf, tb) if verilog else 'no_code'
            by_prob[p].append(verd)
            f.write(json.dumps({'prob': p, 'seed': s, 'verdict': verd, 'verilog': verilog}) + '\n')
            f.flush()
            n_done += 1
            if n_done % 5 == 0:
                el = time.time() - t0; rate = n_done / el if el else 0
                eta = (len(jobs) - n_done) / rate if rate else 0
                sys.stderr.write(f'  [{n_done}/{len(jobs)}] rate={rate:.2f}/s eta={eta:.0f}s last={p} s={s} v={verd}\n')
    f.close()

    p1=p5=0; np_=0
    for p in probs:
        vs = by_prob[p]; n=len(vs); c=sum(1 for x in vs if x=='pass')
        if n==0: continue
        np_+=1; p1+=passk(c,n,1)*100; p5+=passk(c,n,min(5,n))*100
    print(f'\nChipBench framework lowT temp={TEMPERATURE} (NO filters yet):')
    print(f'  pass@1={p1/np_:.1f}%  pass@5={p5/np_:.1f}%')


if __name__ == '__main__': main()
