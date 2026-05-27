"""VEv2 paper-fair: chipmate framework V+P+xverify with NO PERTURBATION.

Identical to vev2_framework.py except:
- No `build_rename_map` / no port renames at all
- No module rename (TopModule kept as-is)
- temp=1.0 (paper-stated)
- Scores against the SAME synth TB (RefModule vs DUTModule random stimulus)

This is the paper-fair baseline. Result expected to be high (memorization).
Subsequent strict-TB enforcement / bug fixes paper-fairly bring down to paper.
"""
import os, sys, json, re, math, time, subprocess, tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/ssd2/yichen/ChipMATE')
from chipmate.backends import OpenAICompatBackend
from chipmate.inference import run_problem

INPUT = '/ssd2/yichen/ChipMATE/examples/verilogeval_v2_reworded_v2.jsonl'
OUT_DIR = '/ssd2/yichen/vev2_framework_reworded_v2'
OUT_SAMPLES = f'{OUT_DIR}/v9b_samples.jsonl'

V_URL = 'http://localhost:8001/v1'
P_URL = 'http://localhost:8002/v1'
V_MODEL = 'core12345/ChipMATE-V-9B'
P_MODEL = 'core12345/ChipMATE-P-9B'

N_SAMPLES = 5
N_INNER = 5
MAX_TURNS = 3
NUM_VERIFY_TESTS = 30
TEMPERATURE = 1.0
PARALLEL_PROBLEMS = 4

RESERVED = {'clk', 'clock', 'rst', 'rst_n', 'reset', 'reset_n', 'aresetn', 'arst'}


def parse_ports(ref_sv):
    m = re.search(r'module\s+\w+\s*\((.*?)\)\s*;', ref_sv, re.DOTALL)
    if not m: return []
    ports = []
    for tok in re.finditer(
        r'(input|output|inout)\s+(?:reg|wire|logic)?\s*(?:signed\s+)?(?:\[[^\]]+\]\s*)?(\w+)',
        m.group(1)):
        ports.append((tok.group(1), tok.group(2)))
    return ports


def synth_tb(ref_sv):
    """Generate stimulus TB. RefModule vs DUTModule, random stimulus."""
    ports = parse_ports(ref_sv)
    if not ports: return None
    inputs  = [(d,n) for d,n in ports if d == 'input']
    outputs = [(d,n) for d,n in ports if d == 'output']
    has_clk = any(n in ('clk','clock') for _,n in inputs)
    def width(n):
        m = re.search(rf'(?:input|output|inout)\s+(?:reg|wire|logic)?\s*(?:signed\s+)?(\[[^\]]+\])?\s*{re.escape(n)}\b', ref_sv)
        if m and m.group(1):
            parts = m.group(1).strip('[] ').split(':')
            try: return abs(int(parts[0])-int(parts[1]))+1
            except: pass
        return 1
    lines = ['`timescale 1ns/1ps', 'module tb();']
    if has_clk: lines += ['reg clk=0;', 'initial forever #5 clk=~clk;']
    for _,n in inputs:
        if n in ('clk','clock'): continue
        lines.append(f'reg [{width(n)-1}:0] {n};')
    for _,n in outputs:
        lines.append(f'wire [{width(n)-1}:0] {n}_ref, {n}_dut;')
    def conn(suffix=''):
        cs = []
        for d,n in ports:
            if n in ('clk','clock'): cs.append(f'.{n}(clk)')
            elif d == 'output': cs.append(f'.{n}({n}{suffix})')
            else: cs.append(f'.{n}({n})')
        return ', '.join(cs)
    lines += [f'RefModule ref_inst({conn("_ref")});', f'DUTModule dut_inst({conn("_dut")});']
    lines += ['integer errors=0, total=0, i;', 'initial begin']
    for _,n in inputs:
        if n in ('rst_n','reset_n','aresetn'): lines += [f'  {n}=0;']
        elif n in ('rst','reset','arst'): lines += [f'  {n}=1;']
        elif n in ('clk','clock'): pass
        else: lines += [f'  {n}=0;']
    if has_clk: lines += ['  @(posedge clk); @(posedge clk);']
    for _,n in inputs:
        if n in ('rst_n','reset_n','aresetn'): lines += [f'  {n}=1;']
        elif n in ('rst','reset','arst'): lines += [f'  {n}=0;']
    if has_clk: lines += ['  @(posedge clk);']
    lines += ['  for (i=0; i<200; i=i+1) begin']
    for _,n in inputs:
        if n in RESERVED: continue
        lines += [f'    {n} = $random;']
    if has_clk: lines += ['    @(posedge clk);', '    #1;']
    else: lines += ['    #1;']
    cond = ' || '.join(f'({n}_ref !== {n}_dut)' for _,n in outputs)
    lines += [f'    total = total + 1;', f'    if ({cond}) errors = errors + 1;', '  end']
    lines += ['  $display("Mismatches: %0d in %0d samples", errors, total);',
              '  if (errors==0) $display("STRICT_PASS");',
              '  else $display("STRICT_FAIL (%0d)", errors);', '  $finish;', 'end',
              'initial begin', '  #500000; $display("TIMEOUT");',
              '  $display("Mismatches: 9999 in 0 samples"); $finish;', 'end', 'endmodule']
    return '\n'.join(lines)


def score(candidate, ref, tb):
    if not candidate or 'module' not in candidate: return 'no_code'
    # rename candidate's first module to DUTModule, ref's to RefModule
    code = re.sub(r'\bmodule\s+\w+', 'module DUTModule', candidate, count=1)
    refr = re.sub(r'\bmodule\s+\w+', 'module RefModule', ref, count=1)
    with tempfile.TemporaryDirectory() as td:
        open(f'{td}/dut.sv','w').write(code)
        open(f'{td}/ref.sv','w').write(refr)
        open(f'{td}/tb.sv','w').write(tb)
        try:
            r = subprocess.run(['iverilog','-g2012','-o','s.vvp','-s','tb','dut.sv','ref.sv','tb.sv'],
                              cwd=td, capture_output=True, text=True, timeout=30)
            if r.returncode != 0: return 'compile_fail'
            sim = subprocess.run(['vvp','s.vvp'], cwd=td, capture_output=True, text=True, timeout=30)
            out = sim.stdout
            if 'STRICT_PASS' in out: return 'pass'
            m = re.search(r'Mismatches:\s*(\d+)\s+in\s+(\d+)', out)
            if m and int(m.group(1))==0 and int(m.group(2))>0: return 'pass'
            return 'mismatch'
        except subprocess.TimeoutExpired: return 'timeout'


def passk(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def gen_one(task_id, question, ref_sv, v, p, seed):
    try:
        r = run_problem(task_id=task_id, question=question, ref_sv=ref_sv,
                        v_backend=v, p_backend=p, n=N_INNER, max_turns=MAX_TURNS,
                        num_verify_tests=NUM_VERIFY_TESTS, temperature=TEMPERATURE,
                        seed=seed)
        return r.verilog
    except Exception as e:
        sys.stderr.write(f'[{task_id} s={seed}] {e}\n')
        return ''


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    probs = [json.loads(l) for l in open(INPUT)]
    print(f'VEv2 framework REWORDED_V2 (no perturb, temp={TEMPERATURE}): {len(probs)} problems × {N_SAMPLES} samples', file=sys.stderr)

    v = OpenAICompatBackend(model=V_MODEL, api_key='EMPTY', base_url=V_URL)
    p = OpenAICompatBackend(model=P_MODEL, api_key='EMPTY', base_url=P_URL)

    pre = {}
    for prob in probs:
        tb = synth_tb(prob['ref_sv'])
        pre[prob['task_id']] = (prob['question'], prob['ref_sv'], tb)

    done = set()
    if os.path.exists(OUT_SAMPLES):
        for l in open(OUT_SAMPLES):
            try: r=json.loads(l); done.add((r['task_id'], r['seed']))
            except: pass

    jobs = []
    for prob in probs:
        if not pre[prob['task_id']][2]: continue
        for s in range(N_SAMPLES):
            if (prob['task_id'], s) in done: continue
            jobs.append((prob['task_id'], s))
    print(f'  {len(jobs)} jobs todo (resume {len(done)})', file=sys.stderr)

    f = open(OUT_SAMPLES, 'a')
    by_prob = defaultdict(list)
    if os.path.exists(OUT_SAMPLES):
        for l in open(OUT_SAMPLES):
            try: r=json.loads(l); by_prob[r['task_id']].append(r['verdict'])
            except: pass

    t0 = time.time(); n_done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_PROBLEMS) as pool:
        futs = {pool.submit(gen_one, tid, pre[tid][0], pre[tid][1], v, p, s): (tid, s) for (tid, s) in jobs}
        for fut in as_completed(futs):
            tid, s = futs[fut]
            try: verilog = fut.result() or ''
            except: verilog = ''
            verd = score(verilog, pre[tid][1], pre[tid][2]) if verilog else 'no_code'
            by_prob[tid].append(verd)
            f.write(json.dumps({'task_id': tid, 'seed': s, 'verdict': verd, 'verilog': verilog}) + '\n')
            f.flush()
            n_done += 1
            if n_done % 10 == 0:
                el = time.time() - t0; rate = n_done / el
                eta = (len(jobs) - n_done) / rate if rate else 0
                sys.stderr.write(f'  [{n_done}/{len(jobs)}] rate={rate:.2f}/s eta={eta:.0f}s last={tid} s={s} v={verd}\n')
    f.close()

    p1=p5=0; np_=0
    for prob in probs:
        vs = by_prob.get(prob['task_id'], [])
        if not vs: continue
        n=len(vs); c=sum(1 for x in vs if x=='pass')
        np_+=1; p1+=passk(c,n,1)*100; p5+=passk(c,n,min(5,n))*100
    print(f'\nVEv2 framework REWORDED_V2 (no perturb, temp=1.0):')
    print(f'  pass@1={p1/np_:.1f}%  pass@5={p5/np_:.1f}%')
    print(f'  paper Agents-9B: 80.1 / 82.4')


if __name__ == '__main__': main()
