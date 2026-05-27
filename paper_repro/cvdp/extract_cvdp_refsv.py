"""Extract REAL port widths per CVDP problem from already-generated candidates.

Earlier cvdp_framework.py defaulted all port widths to 32-bit in the synth
ref_sv, which made chipmate.cross_verify generate stimuli that didn't match the
candidate's actual port sizes — so match_rate stayed near zero and the framework
selection/refinement loop got garbage feedback.

This pass uses the v3 candidates as a port-signature source:
  1. For each problem, find the first candidate that parses cleanly
  2. Extract its module header: parameter defaults + port decls (direction +
     width expression + name)
  3. Evaluate width expressions using the parameter defaults the candidate
     itself declared (e.g. `[N*IN_WIDTH-1:0]` with N=4, IN_WIDTH=4 → 16-bit)
  4. Emit a synthetic ref_sv per problem with CONCRETE widths, so chipmate's
     `parse_ports` returns the right bit-widths and stimuli match

Output: JSON `{problem_id: ref_sv_text}` for cvdp_framework.py to load.
"""
import json, re, sys, os
from collections import defaultdict


def parse_parameters(header_blob):
    """Return {name: int_value} from #(parameter NAME = VAL, ...) decls."""
    params = {}
    # capture #(...) parameter block if present
    m = re.search(r'#\s*\((.*?)\)\s*\(', header_blob, re.DOTALL)
    if not m: return params
    body = m.group(1)
    # 'parameter NAME = VAL' or 'NAME = VAL'
    for pm in re.finditer(r'(?:parameter\s+)?(\w+)\s*=\s*([^,)]+)', body):
        name = pm.group(1)
        val_expr = pm.group(2).strip()
        # numeric or referring to earlier params
        try:
            val = eval(val_expr, {"__builtins__": {}}, dict(params))
            params[name] = int(val)
        except Exception:
            pass
    return params


def parse_ports(header_blob, params):
    """Return list of (direction, width_int, name)."""
    # port list = text between the LAST '(' and matching ')'; we approximate
    # by finding `\) *;` after the last `(` in the module header
    # Header form: `module NAME #(...) ( ports );` or `module NAME ( ports );`
    m = re.search(r'module\s+\w+\s*(?:#\s*\([^)]*\))?\s*\((.*?)\)\s*;',
                  header_blob, re.DOTALL)
    if not m: return []
    body = m.group(1)
    ports = []
    for tok in re.finditer(
        r'\b(input|output|inout)\s+(?:reg|wire|logic)?\s*(?:signed\s+)?(\[[^\]]+\])?\s*(\w+)',
        body):
        direction, w_expr, name = tok.group(1), tok.group(2), tok.group(3)
        w = 1
        if w_expr:
            inner = w_expr.strip('[] ')
            parts = inner.split(':')
            if len(parts) == 2:
                try:
                    hi = eval(parts[0].strip(), {"__builtins__": {}}, dict(params))
                    lo = eval(parts[1].strip(), {"__builtins__": {}}, dict(params))
                    w = abs(int(hi) - int(lo)) + 1
                except Exception:
                    w = 32  # fall back
        ports.append((direction, w, name))
    return ports


def extract_verilog_from_completion(c):
    """Pull bare verilog out of `\`\`\`verilog ... \`\`\`` wrapped completion."""
    if not c: return ''
    m = re.search(r'```(?:verilog|systemverilog|sv)?\s*\n(.*?)```', c, re.DOTALL)
    return m.group(1).strip() if m else c


def synth_ref_sv(toplevel, ports):
    if not ports: return None
    lines = [f'module {toplevel}(']
    decls = []
    for direction, width, name in ports:
        if width <= 1:
            decls.append(f'  {direction} {name}')
        else:
            decls.append(f'  {direction} [{width-1}:0] {name}')
    lines.append(',\n'.join(decls))
    lines.append(');')
    lines.append('endmodule')
    return '\n'.join(lines)


def main():
    bench_path = '/home/yichen/workspace/LLaMA-Factory/data/cvdp_bench_verilog_VTRACK_cid003.jsonl'
    rollout_paths = [
        '/ssd2/yichen/paper_eval/rollouts/chipmate-9b-framework-v3__cvdp_cid003.jsonl',
        '/ssd2/yichen/paper_eval/rollouts/chipmate-9b-framework-v2__cvdp_cid003.jsonl',
        '/ssd2/yichen/paper_eval/rollouts/chipmate-9b-framework__cvdp_cid003.jsonl',
    ]
    out_path = '/ssd2/yichen/paper_eval/cvdp_refsv.json'

    bench = {}
    with open(bench_path) as f:
        for line in f:
            r = json.loads(line)
            bench[r['problem_id']] = r

    # Collect all candidates per problem
    by_pid = defaultdict(list)
    for p in rollout_paths:
        if not os.path.exists(p): continue
        for line in open(p):
            r = json.loads(line)
            for s in r.get('samples', []):
                c = extract_verilog_from_completion(s.get('completion', ''))
                if c: by_pid[r['problem_id']].append(c)

    out = {}
    n_ok = n_fail = 0
    for pid, row in bench.items():
        toplevel = row['toplevel']
        candidates = by_pid.get(pid, [])
        chosen = None
        for c in candidates:
            params = parse_parameters(c)
            ports = parse_ports(c, params)
            if not ports: continue
            # Pick candidate that has both inputs AND outputs (avoids weird stubs)
            has_in = any(d == 'input' for d,_,_ in ports)
            has_out = any(d == 'output' for d,_,_ in ports)
            if has_in and has_out:
                chosen = ports
                break
        if not chosen and candidates:
            params = parse_parameters(candidates[0])
            ports = parse_ports(candidates[0], params)
            if ports: chosen = ports
        if chosen:
            out[pid] = synth_ref_sv(toplevel, chosen)
            n_ok += 1
        else:
            n_fail += 1

    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'extracted ref_sv for {n_ok}/{n_ok+n_fail} problems', file=sys.stderr)
    print(f'wrote {out_path}', file=sys.stderr)
    # Sample
    for pid in list(out.keys())[:3]:
        print(f'\n--- {pid} ---')
        print(out[pid])


if __name__ == '__main__': main()
