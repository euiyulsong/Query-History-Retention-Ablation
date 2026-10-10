#!/usr/bin/env python3
"""100-example paired ablation with GPT-6 Luna via OpenAI Responses API.

HotpotQA -> two Luna rewrites -> dense retrieval -> Luna document selection
-> freeze identical evidence -> compare docs-only versus queries+docs reasoning.
"""
from __future__ import annotations

import argparse
import os
import json
import random
import re
import string
import time
from collections import Counter
from pathlib import Path


def dump(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = ''.join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def answer_metrics(pred: str, gold: str) -> tuple[float, float]:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    em = float(p == g)
    if p in {'yes', 'no', 'noanswer'} or g in {'yes', 'no', 'noanswer'}:
        return em, em
    a, b = Counter(p.split()), Counter(g.split())
    overlap = sum((a & b).values())
    if not overlap:
        return em, 0.0
    prec = overlap / max(1, sum(a.values()))
    rec = overlap / max(1, sum(b.values()))
    return em, 2 * prec * rec / (prec + rec)


def get_docs(record: dict) -> list[dict]:
    """Hugging Face datasets sometimes presents Sequence(struct) columnwise."""
    ctx = record['context']
    if isinstance(ctx, dict):
        pairs = zip(ctx['title'], ctx['sentences'])
    else:
        pairs = ((x['title'], x['sentences']) for x in ctx)
    out = []
    for k, (title, sentences) in enumerate(pairs):
        out.append({'id': f'D{k:02d}', 'title': title,
                    'text': ' '.join(sentences).strip()})
    return out


def supporting_titles(record: dict) -> list[str]:
    x = record['supporting_facts']
    if isinstance(x, dict):
        vals = x['title']
    else:
        vals = [e['title'] for e in x]
    return list(dict.fromkeys(vals))


def sample_dataset(path: Path, n: int, seed: int) -> list[dict]:
    if path.exists():
        rows = read(path)
        if len(rows) != n:
            raise RuntimeError('Saved sample length differs from --n; use another --out')
        return rows
    from datasets import load_dataset
    allrows = load_dataset('hotpotqa/hotpot_qa', 'distractor', split='validation')
    r = random.Random(seed)
    bridge = [i for i, x in enumerate(allrows) if x['type'] == 'bridge']
    comparison = [i for i, x in enumerate(allrows) if x['type'] == 'comparison']
    nbridge = n // 2
    ncomp = n - nbridge
    if nbridge > len(bridge) or ncomp > len(comparison):
        raise ValueError('Not enough examples for specified n')
    indices = r.sample(bridge, nbridge) + r.sample(comparison, ncomp)
    r.shuffle(indices)
    rows = []
    for idx in indices:
        e = allrows[idx]
        rows.append({'id': e['id'], 'question': e['question'],
                     'answer': e['answer'], 'type': e['type'],
                     'level': e['level'], 'docs': get_docs(e),
                     'support_titles': supporting_titles(e)})
    dump(path, rows)
    return rows


def to_responses_input(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert a *frozen* Chat Completions tool trace to Responses input items.

    No tool execution happens here. This is a controlled counterfactual replay:
    the underlying results are identical in the two arms.
    """
    instructions = []
    items = []
    for msg in messages:
        role = msg['role']
        if role in ('system', 'developer'):
            instructions.append(str(msg.get('content') or ''))
        elif role in ('user', 'assistant'):
            if msg.get('content'):
                items.append({'role': role, 'content': msg['content']})
            for call in msg.get('tool_calls', []):
                items.append({
                    'type': 'function_call',
                    'call_id': call['id'],
                    'name': call['function']['name'],
                    'arguments': call['function']['arguments'],
                })
        elif role == 'tool':
            items.append({
                'type': 'function_call_output',
                'call_id': msg['tool_call_id'],
                'output': str(msg['content']),
            })
        else:
            raise ValueError(f'Unknown message role: {role}')
    return '\n\n'.join(instructions), items


class Luna:
    """OpenAI Responses API client. No vLLM or chat_template_kwargs."""
    def __init__(self, url: str | None, model: str, timeout: float,
                 light_effort: str = 'none', final_effort: str = 'medium',
                 client=None):
        if client is None:
            from openai import OpenAI
            key = os.environ.get('OPENAI_API_KEY')
            if not key:
                raise RuntimeError('Set OPENAI_API_KEY before running this experiment.')
            kw = {'api_key': key, 'timeout': timeout, 'max_retries': 2}
            if url:
                kw['base_url'] = url
            client = OpenAI(**kw)
        self.client = client
        self.model = model
        self.light_effort = light_effort
        self.final_effort = final_effort

    def ask_messages(self, messages: list[dict], max_tokens: int,
                     thinking: bool = False, output_format: dict | None = None) -> dict:
        instructions, items = to_responses_input(messages)
        t = time.perf_counter()
        kwargs = dict(
            model=self.model,
            instructions=instructions,
            input=items,
            reasoning={'effort': self.final_effort if thinking else self.light_effort},
            max_output_tokens=max_tokens,
            store=False,
        )
        if output_format is not None:
            kwargs['text'] = {'format': output_format}
        rsp = self.client.responses.create(**kwargs)
        content = rsp.output_text or ''
        usage = rsp.usage
        status = getattr(rsp, 'status', None)
        incomplete = getattr(rsp, 'incomplete_details', None)
        truncated = (status == 'incomplete' and
                     getattr(incomplete, 'reason', None) == 'max_output_tokens')
        return {
            'text': content.strip(),
            'seconds': round(time.perf_counter() - t, 3),
            'prompt_tokens': getattr(usage, 'input_tokens', None),
            'completion_tokens': getattr(usage, 'output_tokens', None),
            'finish_reason': 'length' if truncated else str(status or 'unknown'),
            'response_status': status,
        }

    def ask(self, system: str, user: str, max_tokens: int,
            thinking: bool = False, output_format: dict | None = None) -> dict:
        return self.ask_messages(
            [{'role': 'system', 'content': system},
             {'role': 'user', 'content': user}],
            max_tokens=max_tokens, thinking=thinking, output_format=output_format)


def clean_query(raw: str, fallback: str) -> str:
    s = raw.strip()
    s = re.sub(r'^```[^\n]*\n|\n```$', '', s).strip()
    s = s.splitlines()[0].strip() if s else ''
    s = re.sub(r'^(query|search query|q\d+)\s*:\s*', '', s, flags=re.I)
    s = s.strip(' "\'`')
    return s[:400] if s else fallback


def parse_indices(text: str, count: int, max_keep: int) -> list[int] | None:
    """Strict selection parsing; invalid output is logged as a fallback."""
    found = re.search(r'\{[^{}]*"selected_indices"\s*:\s*\[[^\]]*\][^{}]*\}', text, re.S)
    candidate = found.group() if found else text
    try:
        arr = json.loads(candidate)['selected_indices']
        if not isinstance(arr, list):
            return None
        if any(type(i) is not int or i < 0 or i >= count for i in arr):
            return None
        ids = list(dict.fromkeys(arr))
        return ids[:max_keep] if ids else None
    except (TypeError, ValueError, KeyError):
        return None


SELECTOR_SCHEMA = {
    'type': 'json_schema',
    'name': 'document_selection',
    'strict': True,
    'schema': {
        'type': 'object',
        'properties': {
            'selected_indices': {
                'type': 'array',
                'items': {'type': 'integer'},
            }
        },
        'required': ['selected_indices'],
        'additionalProperties': False,
    },
}


def select_docs(llm: Luna, question: str, candidates: list[dict],
                keep: int, label: str, max_chars: int) -> tuple[list[dict], dict]:
    if not candidates or keep < 1:
        raise RuntimeError(f'{label}: no selection candidates or keep < 1')
    shown = '\n\n'.join(
        f'[{i}] Title: {d["title"]}\nText: {d["text"][:max_chars]}'
        for i, d in enumerate(candidates)
    )
    instruction = (
        'You are a retrieval context selector. Select between one and '
        f'{keep} document indices from the candidates sufficient to answer '
        'the QUESTION, including both hops when necessary. Do not answer '
        'the question. Output selected_indices as an array of zero-based '
        'integers with no duplicates.'
    )
    # Structured Outputs prevents a plain-English or malformed JSON response.
    # Never silently substitute the first-k dense passages: the benchmark
    # explicitly evaluates an LLM selector.
    rsp = llm.ask(
        instruction, f'QUESTION: {question}\n\nCANDIDATES:\n{shown}',
        max_tokens=256, thinking=False, output_format=SELECTOR_SCHEMA)
    indices = parse_indices(rsp['text'], len(candidates), keep)
    if indices is None:
        raise RuntimeError(
            f'{label}: invalid LLM selection; status={rsp.get("response_status")} '
            f'raw={rsp["text"][:350]!r}. This sample was NOT silently fallback-selected.')
    out = [candidates[i] for i in indices]
    return out, {
        'label': label, 'raw': rsp['text'], 'indices': indices,
        'fallback_to_dense_order': False,
        'seconds': rsp['seconds'],
        'prompt_tokens': rsp['prompt_tokens'],
        'completion_tokens': rsp['completion_tokens'],
    }


class DenseRetriever:
    def __init__(self, model_name: str, rows: list[dict], device: str):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.np = np
        self.encoder = SentenceTransformer(model_name, device=device)
        self.docs = {r['id']: r['docs'] for r in rows}
        self.embs = {}
        # Separate closed pool per question. DO NOT use answer/supporting titles to retrieve.
        for ri, r in enumerate(rows):
            passages = [d['title'] + '. ' + d['text'] for d in r['docs']]
            self.embs[r['id']] = self.encoder.encode(
                passages, normalize_embeddings=True, convert_to_numpy=True,
                batch_size=16, show_progress_bar=False)
            if (ri + 1) % 25 == 0:
                print(f'Embedded {ri+1}/{len(rows)} question-local corpora', flush=True)

    def search(self, record_id: str, query: str, topk: int) -> list[dict]:
        # BGE v1.5 short query -> passage recommended prefix.
        q = self.encoder.encode(
            ['Represent this sentence for searching relevant passages: ' + query],
            normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.embs[record_id] @ q
        order = self.np.argsort(-scores)[:topk]
        return [dict(self.docs[record_id][int(i)], score=float(scores[i]))
                for i in order]


def build_trace(llm: Luna, dense: DenseRetriever, record: dict,
                retrieve_k: int, step1_keep: int, final_keep: int,
                selector_chars: int) -> dict:
    question = record['question']
    q1out = llm.ask(
        'Write ONE concise English web/document SEARCH QUERY to help answer '
        'this multi-hop question. Output one line only, no answer.',
        question, max_tokens=120)
    q1 = clean_query(q1out['text'], question)
    hit1 = dense.search(record['id'], q1, retrieve_k)
    selected1, select1log = select_docs(
        llm, question, hit1, step1_keep, 'first_step', selector_chars)
    evidence1 = '\n\n'.join(
        f'[{d["id"]}] {d["title"]}: {d["text"][:selector_chars]}'
        for d in selected1)
    q2out = llm.ask(
        'You generate the SECOND and FINAL SEARCH QUERY for a two-hop QA '
        'task. Examine the original question and first retrieved evidence. '
        'Search for missing information needed to answer the question. '
        'Avoid merely repeating the first query. Output ONLY one search query '
        'on one line. Never answer the question.',
        f'QUESTION: {question}\nFIRST QUERY: {q1}\nFIRST EVIDENCE:\n{evidence1}',
        max_tokens=140)
    q2 = clean_query(q2out['text'], question)
    hit2 = dense.search(record['id'], q2, retrieve_k)

    # Select a SECOND, complementary group; q2 actually saw selected1.
    # The final official tool transcript can now faithfully replay both steps.
    prior_ids = {d['id'] for d in selected1}
    step2_candidates = [d for d in hit2 if d['id'] not in prior_ids]
    step2_question = (
        question + '\nAlready retained from the first step: '
        + '; '.join(d['title'] for d in selected1)
        + '\nSelect complementary information for the remaining hop.'
    )
    selected2, select2log = select_docs(
        llm, step2_question, step2_candidates,
        min(step1_keep, max(1, final_keep - len(selected1))), 'second_step', selector_chars)
    selected = [*selected1, *selected2]
    for d in selected1:
        d['retrieved_in_steps'] = [1]
    for d in selected2:
        d['retrieved_in_steps'] = [2]

    gold = set(record['support_titles'])
    selected_titles = {d['title'] for d in selected}
    retrieved_titles = {d['title'] for d in hit1 + hit2}
    return {
        'id': record['id'], 'q1': q1, 'q2': q2,
        'q1_log': q1out, 'q2_log': q2out,
        'hit1_ids': [d['id'] for d in hit1],
        'hit2_ids': [d['id'] for d in hit2],
        'selected1_ids': [d['id'] for d in selected1],
        'selected2_ids': [d['id'] for d in selected2],
        'selected': selected, 'select1_log': select1log,
        'select2_log': select2log,
        'gold_support_titles': sorted(gold),
        'retrieved_gold_support_recall': len(gold & retrieved_titles) / max(1, len(gold)),
        'selected_gold_support_recall': len(gold & selected_titles) / max(1, len(gold)),
        'selected_all_support': gold <= selected_titles,
    }


def build_final_prompt(record: dict, trace: dict, include_queries: bool,
                       final_chars: int) -> str:
    # Critical paired control: always the SAME selected docs, SAME order,
    # SAME original question, SAME formatting aside from actual query lines.
    lines = [f'ORIGINAL QUESTION:\n{record["question"]}', '', 'RETRIEVAL HISTORY:']
    for step in (1, 2):
        lines.append(f'[Search step {step}]')
        if include_queries:
            lines.append(f'Search query: {trace[f"q{step}"]}')
        docs_here = [d['id'] for d in trace['selected']
                     if step in d['retrieved_in_steps']]
        lines.append('Evidence IDs returned and retained: ' +
                     (', '.join(docs_here) if docs_here else '(none)'))
    lines.extend(['', 'SELECTED EVIDENCE:'])
    for d in trace['selected']:
        lines.append(f'[{d["id"]}] {d["title"]}\n{d["text"][:final_chars]}')
    lines.extend(['', 'Return your answer using the evidence.'])
    return '\n\n'.join(lines)


ANSWER_SYSTEM = (
    'You are a careful multi-hop question-answering model. '
    'Answer the ORIGINAL QUESTION using only SELECTED EVIDENCE. '
    'Search query strings, when shown, describe prior retrieval intent and '
    'are NEVER factual evidence. Do not rely on them for facts. '
    'Reason through necessary connections and give a SHORT factoid answer '
    'or yes/no as appropriate. If the evidence does not support an answer, '
    'say unknown. Output exactly one line: FINAL_ANSWER: <answer>.'
)


def parse_answer(s: str) -> str:
    matches = re.findall(r'FINAL_ANSWER\s*:\s*([^\n]+)', s, re.I)
    if matches:
        return matches[-1].strip().strip('"\'`')
    if not s.strip():
        return ''
    return s.strip().splitlines()[-1].strip()


def build_tool_messages(record: dict, trace: dict, include_queries: bool,
                        final_chars: int) -> list[dict]:
    """Faithful Chat Completions tool transcript; args are the ONLY arm difference.

    This is an ablation/replay, not a claim that the assistant actually called
    a tool with an empty query in the docs-only arm.
    """
    messages = [
        {'role': 'system', 'content': ANSWER_SYSTEM},
        {'role': 'user', 'content': record['question']}
    ]
    for step in (1, 2):
        call_id = f'call_replay_{step}'
        arg = trace[f'q{step}'] if include_queries else ''
        messages.append({
            'role': 'assistant',
            'content': None,
            'tool_calls': [{
                'id': call_id,
                'type': 'function',
                'function': {'name': 'search_docs',
                             'arguments': json.dumps({'query': arg}, ensure_ascii=False)}
            }]
        })
        docs = [d for d in trace['selected'] if step in d['retrieved_in_steps']]
        evidence = [{
            'id': d['id'], 'title': d['title'], 'text': d['text'][:final_chars]
        } for d in docs]
        messages.append({
            'role': 'tool', 'tool_call_id': call_id,
            'content': json.dumps({'documents': evidence}, ensure_ascii=False)
        })
    return messages


def judge(llm: Luna, record: dict, trace: dict, arm: str,
          max_tokens: int, thinking: bool, final_chars: int,
          context_format: str = 'tool_messages') -> dict:
    with_queries = arm == 'query_plus_docs'
    if context_format == 'tool_messages':
        messages = build_tool_messages(record, trace, with_queries, final_chars)
        rsp = llm.ask_messages(messages, max_tokens=max_tokens, thinking=thinking)
    elif context_format == 'text':
        prompt = build_final_prompt(record, trace, with_queries, final_chars)
        rsp = llm.ask(ANSWER_SYSTEM, prompt, max_tokens=max_tokens, thinking=thinking)
    else:
        raise ValueError(f'Unknown context_format {context_format}')
    pred = parse_answer(rsp['text'])
    em, f1 = answer_metrics(pred, record['answer'])
    return {'answer': pred, 'em': em, 'f1': f1, 'seconds': rsp['seconds'],
            'prompt_tokens': rsp['prompt_tokens'],
            'completion_tokens': rsp['completion_tokens'],
            'finish_reason': rsp['finish_reason'],
            'response_status': rsp['response_status'],
            'answer_words': len(pred.split()),
            'format_ok': bool(re.search(r'FINAL_ANSWER\s*:', rsp['text'], re.I)),
            'raw': rsp['text']}


def paired_bootstrap(deltas: list[float], seed: int, iters: int = 5000):
    if not deltas:
        return [None, None]
    import numpy as np
    r = np.random.default_rng(seed)
    xs = np.asarray(deltas, dtype=float)
    draws = r.choice(xs, (iters, len(xs)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def mean_nonnull(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def summarize(rows: list[dict], traces: dict, outputs: dict, seed: int) -> dict:
    paired = [r for r in rows if r['id'] in outputs and
              all(arm in outputs[r['id']] for arm in ['docs_only', 'query_plus_docs'])]
    obj = {'requested_n': len(rows), 'paired_n': len(paired),
           'missing_ids': [r['id'] for r in rows if r not in paired and
                           (r['id'] not in outputs or
                            len(outputs.get(r['id'], {})) != 2)],
           'selected_gold_support_recall': mean_nonnull(
               [traces[r['id']]['selected_gold_support_recall']
                for r in paired if r['id'] in traces]),
           'retrieved_gold_support_recall': mean_nonnull(
               [traces[r['id']]['retrieved_gold_support_recall']
                for r in paired if r['id'] in traces]),
           'selected_all_support_rate': mean_nonnull(
               [float(traces[r['id']]['selected_all_support'])
                for r in paired if r['id'] in traces]),
           'selector_fallback_total': sum(
               int(traces[r['id']][field]['fallback_to_dense_order'])
               for r in paired if r['id'] in traces
               for field in ['select1_log', 'select2_log']),
           'by_type': {}}
    for subset_name, subset in [('all', paired),
                                ('bridge', [r for r in paired if r['type'] == 'bridge']),
                                ('comparison', [r for r in paired if r['type'] == 'comparison'])]:
        d = {'n': len(subset)}
        for arm in ('docs_only', 'query_plus_docs'):
            scores = [outputs[r['id']][arm] for r in subset]
            d[arm] = {k: mean_nonnull([v[k] for v in scores])
                      for k in ('em', 'f1', 'seconds', 'prompt_tokens',
                                'completion_tokens')}
            d[arm]['empty_answers'] = sum(not v['answer'] for v in scores)
            d[arm]['finish_reason_length'] = sum(
                v['finish_reason'] == 'length' for v in scores)
            d[arm]['long_answer_over_8_words'] = sum(
                len(v['answer'].split()) > 8 for v in scores)
            d[arm]['format_failures'] = sum(
                ('FINAL_ANSWER:' not in v.get('raw', 'FINAL_ANSWER:')) for v in scores)
        for metric in ('em', 'f1'):
            changes = [outputs[r['id']]['query_plus_docs'][metric] -
                       outputs[r['id']]['docs_only'][metric] for r in subset]
            d[f'delta_{metric}'] = mean_nonnull(changes)
            d[f'delta_{metric}_95ci'] = paired_bootstrap(changes, seed)
            d[f'{metric}_wins'] = sum(x > 1e-9 for x in changes)
            d[f'{metric}_losses'] = sum(x < -1e-9 for x in changes)
            d[f'{metric}_ties'] = sum(abs(x) <= 1e-9 for x in changes)
        obj['by_type'][subset_name] = d
    return obj


def save_results(outdir: Path, rows: list[dict], traces: dict,
                 outputs: dict, seed: int):
    summary = summarize(rows, traces, outputs, seed)
    dump(outdir / 'summary.json', summary)
    with (outdir / 'predictions.jsonl').open('w', encoding='utf-8') as f:
        for r in rows:
            rid = r['id']
            f.write(json.dumps({
                'id': rid, 'question': r['question'], 'gold': r['answer'],
                'type': r['type'],
                'q1': traces.get(rid, {}).get('q1'),
                'q2': traces.get(rid, {}).get('q2'),
                'selected_docs': [d['id'] for d in traces.get(rid, {}).get('selected', [])],
                'arms': outputs.get(rid, {})}, ensure_ascii=False) + '\n')
    return summary


def print_dry_run(args):
    record = {'question': 'What year was the author of Book A born?'}
    trace = {'q1': 'Book A author name', 'q2': 'author birth year', 'selected': [
        {'id': 'D00', 'title': 'Book A', 'text': 'Book A was written by Alex Kim.', 'retrieved_in_steps': [1]},
        {'id': 'D02', 'title': 'Alex Kim', 'text': 'Alex Kim was born in 1979.', 'retrieved_in_steps': [2]},
    ]}
    for arm, with_queries in [('docs_only', False), ('query_plus_docs', True)]:
        if args.context_format == 'tool_messages':
            instructions, items = to_responses_input(
                build_tool_messages(record, trace, with_queries, args.final_chars))
        else:
            instructions = ANSWER_SYSTEM
            items = [{'role': 'user', 'content': build_final_prompt(
                record, trace, with_queries, args.final_chars)}]
        print(json.dumps({
            'arm': arm, 'model': args.model, 'reasoning_effort': (args.light_effort if args.no_final_thinking else args.final_effort),
            'instructions': instructions, 'input': items,
        }, indent=2, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path, default=Path('results_luna_100'))
    p.add_argument('--base-url', default=None, help='Optional custom Responses endpoint; default OpenAI API')
    p.add_argument('--model', default='gpt-6-luna')
    p.add_argument('--embedding', default='BAAI/bge-base-en-v1.5')
    p.add_argument('--device', default='cpu', help='CPU default keeps GPU for vLLM; can use cuda')
    p.add_argument('--retrieval-topk', type=int, default=8)
    p.add_argument('--step1-keep', type=int, default=2)
    p.add_argument('--final-keep', type=int, default=4)
    p.add_argument('--selector-chars', type=int, default=1300)
    p.add_argument('--final-chars', type=int, default=2000)
    p.add_argument('--answer-max-tokens', type=int, default=3072)
    p.add_argument('--no-final-thinking', action='store_true',
                   help='Use light reasoning effort for answer generation too')
    p.add_argument('--light-effort', choices=['none', 'low', 'medium'], default='none')
    p.add_argument('--final-effort', choices=['none', 'low', 'medium', 'high'], default='medium')
    p.add_argument('--dry-run', action='store_true',
                   help='Print paired Responses request shapes without API/dataset downloads')
    p.add_argument('--context-format', choices=['tool_messages', 'text'],
                   default='tool_messages', help='Official tool transcript or text concatenation')
    p.add_argument('--timeout', type=float, default=240)
    p.add_argument('--continue-on-error', action='store_true')
    args = p.parse_args()
    if args.dry_run:
        print_dry_run(args)
        return
    if not os.environ.get('OPENAI_API_KEY'):
        p.error('OPENAI_API_KEY is missing. Export your API key before running.')
    args.out.mkdir(parents=True, exist_ok=True)
    config_file = args.out / 'config.json'
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if config_file.exists():
        old = read(config_file)
        # Resuming with changed prompts/params would invalidate paired controls.
        checked = ('n', 'seed', 'model', 'embedding', 'retrieval_topk',
                   'step1_keep', 'final_keep', 'selector_chars', 'final_chars',
                   'answer_max_tokens', 'no_final_thinking', 'context_format',
                   'light_effort', 'final_effort', 'base_url')
        for k in checked:
            if old.get(k) != config[k]:
                raise RuntimeError(f'Changed config {k}; use a fresh --out folder')
    else:
        dump(config_file, config)
    rows = sample_dataset(args.out / 'sample.json', args.n, args.seed)
    traces_path = args.out / 'traces.json'
    outputs_path = args.out / 'outputs.json'
    traces = read(traces_path) if traces_path.exists() else {}
    outputs = read(outputs_path) if outputs_path.exists() else {}
    llm = Luna(args.base_url, args.model, args.timeout, args.light_effort,
               args.light_effort if args.no_final_thinking else args.final_effort)
    dense = None
    missing_traces = [r for r in rows if r['id'] not in traces]
    if missing_traces:
        print('Initializing dense encoder...', flush=True)
        dense = DenseRetriever(args.embedding, rows, args.device)
    for i, record in enumerate(rows, start=1):
        rid = record['id']
        try:
            if rid not in traces:
                traces[rid] = build_trace(
                    llm, dense, record, args.retrieval_topk, args.step1_keep,
                    args.final_keep, args.selector_chars)
                dump(traces_path, traces)
            outputs.setdefault(rid, {})
            # Randomize which arm runs first per case to reduce order bias.
            arms = ['docs_only', 'query_plus_docs']
            random.Random(args.seed + int(rid[:8], 16)).shuffle(arms)
            for arm in arms:
                if arm not in outputs[rid]:
                    outputs[rid][arm] = judge(
                        llm, record, traces[rid], arm, args.answer_max_tokens,
                        not args.no_final_thinking, args.final_chars, args.context_format)
                    dump(outputs_path, outputs)
            a, b = outputs[rid]['docs_only'], outputs[rid]['query_plus_docs']
            print(f'[{i:03d}/{len(rows)}] {rid} {record["type"]} '
                  f'docs EM={a["em"]:.0f} F1={a["f1"]:.2f} | '
                  f'+queries EM={b["em"]:.0f} F1={b["f1"]:.2f} '
                  f'gold_selected={traces[rid]["selected_gold_support_recall"]:.2f}',
                  flush=True)
        except Exception as exc:
            print(f'[ERROR] {rid} {type(exc).__name__}: {exc}', flush=True)
            if not args.continue_on_error:
                raise
        if i % 10 == 0 or i == len(rows):
            summary = save_results(args.out, rows, traces, outputs, args.seed)
    print('\nFINAL SUMMARY\n' + json.dumps(summary, indent=2, ensure_ascii=False))
    if summary['paired_n'] < args.n:
        raise SystemExit(f'INCOMPLETE: {summary["paired_n"]}/{args.n} valid pairs. '
                         'Rerun with same --out to resume.')


if __name__ == '__main__':
    main()
