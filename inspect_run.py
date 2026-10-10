#!/usr/bin/env python3
"""Print original final answers and selector responses from a completed run."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('smoke_luna_4')
outs = json.loads((root / 'outputs.json').read_text(encoding='utf-8'))
traces = json.loads((root / 'traces.json').read_text(encoding='utf-8'))
rows = json.loads((root / 'sample.json').read_text(encoding='utf-8'))
for row in rows:
    rid = row['id']
    print('\n' + '=' * 80)
    print('ID:', rid, '| gold:', repr(row['answer']))
    print('QUESTION:', row['question'])
    tr = traces.get(rid)
    if tr:
        for n in (1, 2):
            log = tr.get(f'select{n}_log', {})
            print(f'SELECTOR #{n}: fallback={log.get("fallback_to_dense_order")}, '
                  f'raw={log.get("raw")!r}, selected={log.get("indices")}')
    for name in ('docs_only', 'query_plus_docs'):
        x = outs.get(rid, {}).get(name, {})
        ans = str(x.get('answer', ''))
        raw = str(x.get('raw', ''))
        print(f'{name}: EM={x.get("em")} F1={x.get("f1")}, '
              f'words={len(ans.split())}, output_tokens={x.get("completion_tokens")}, '
              f'status={x.get("finish_reason")}')
        print('  parsed:', repr(ans))
        print('  raw:', repr(raw[:1200]))
