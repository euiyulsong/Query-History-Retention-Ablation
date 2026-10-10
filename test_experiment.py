import unittest
from experiment import (
    answer_metrics, build_final_prompt, build_tool_messages, clean_query, get_docs,
    normalize_answer, parse_answer, parse_indices, supporting_titles,
    summarize,
)


class ExperimentTests(unittest.TestCase):
    def test_hotpot_metrics(self):
        self.assertEqual(answer_metrics('The United States!', 'United States'), (1., 1.))
        self.assertEqual(answer_metrics('yes', 'no'), (0., 0.))
        self.assertEqual(answer_metrics('San Francisco', 'Francisco'), (0., 2/3))

    def test_dataset_columnwise(self):
        rec = {'context': {'title': ['Alpha', 'Beta'], 'sentences': [['Hi.', ' Bye.'], ['foo']]},
               'supporting_facts': {'title': ['Beta', 'Beta', 'Alpha'], 'sent_id': [0, 1, 0]}}
        self.assertEqual(len(get_docs(rec)), 2)
        self.assertEqual(supporting_titles(rec), ['Beta', 'Alpha'])

    def test_selector_is_strict(self):
        self.assertEqual(parse_indices('{"selected_indices":[2,0,2]}', 3, 2), [2, 0])
        self.assertIsNone(parse_indices('{"selected_indices":[5]}', 3, 2))
        self.assertIsNone(parse_indices('{"selected_indices":["1"]}', 3, 2))
        self.assertIsNone(parse_indices('blah', 3, 2))

    def test_prompt_pairs(self):
        record = {'question': 'Where is it?', 'answer': 'there'}
        trace = {'q1': 'location Foo', 'q2': 'Foo headquarters',
                 'selected': [
                     {'id':'D00','title':'A','text':'Location there.', 'retrieved_in_steps':[1]},
                     {'id':'D03','title':'B','text':'Other.', 'retrieved_in_steps':[1,2]},
                 ]}
        a = build_final_prompt(record, trace, False, 2000)
        b = build_final_prompt(record, trace, True, 2000)
        self.assertNotIn('location Foo', a)
        self.assertIn('location Foo', b)
        self.assertIn('Foo headquarters', b)
        for doc in trace['selected']:
            for prompt in (a, b):
                self.assertIn(doc['text'], prompt)
                self.assertIn(doc['id'], prompt)
        msgs_a = build_tool_messages(record, trace, False, 2000)
        msgs_b = build_tool_messages(record, trace, True, 2000)
        self.assertEqual(len(msgs_a), 6)
        for i in (3,5):
            self.assertEqual(msgs_a[i], msgs_b[i])  # Tool results byte-for-byte identical
        self.assertIn('Foo headquarters', msgs_b[4]['tool_calls'][0]['function']['arguments'])
        self.assertNotIn('Foo headquarters', msgs_a[4]['tool_calls'][0]['function']['arguments'])
        self.assertEqual(a.replace('[Search step 1]', '[Search step 1]\n\nSearch query: location Foo')
                         .replace('[Search step 2]', '[Search step 2]\n\nSearch query: Foo headquarters'), b)

    def test_ans_extraction(self):
        self.assertEqual(parse_answer('Thinking...\nFINAL_ANSWER: Sydney'), 'Sydney')
        self.assertEqual(clean_query('Query: bright stars\nexplanation', 'none'), 'bright stars')

    def test_pairwise_summary(self):
        rows = [{'id': 'a', 'type':'bridge'}, {'id':'b','type':'comparison'}]
        traces = {rid: {'selected_gold_support_recall':1., 'retrieved_gold_support_recall':1.,
                        'selected_all_support':True,
                        'select1_log': {'fallback_to_dense_order': False},
                        'select2_log': {'fallback_to_dense_order': False}}
                  for rid in ('a','b')}
        def r(em, f1):
            return {'answer':'A', 'em': em, 'f1':f1,'seconds':1.,'prompt_tokens':100,'completion_tokens':10, 'finish_reason':'stop'}
        outputs = {'a':{'docs_only':r(0.,0.),'query_plus_docs':r(1.,1.)},
                   'b':{'docs_only':r(1.,1.),'query_plus_docs':r(0.,0.)}}
        summary = summarize(rows, traces, outputs, 42)
        self.assertEqual(summary['paired_n'], 2)
        self.assertEqual(summary['by_type']['all']['delta_em'], 0.)
        self.assertEqual(summary['by_type']['all']['em_wins'], 1)
        self.assertEqual(summary['by_type']['all']['em_losses'], 1)




class LunaResponsesTests(unittest.TestCase):
    def test_tool_replay_structure(self):
        from experiment import to_responses_input
        record = {'question': 'Who wrote Book A?'}
        trace = {'q1':'Book A author','q2':'writer hometown', 'selected': [
            {'id':'D01','title':'Book A','text':'Jane wrote Book A.', 'retrieved_in_steps':[1]},
            {'id':'D02','title':'Jane','text':'Jane was raised in Rome.', 'retrieved_in_steps':[2]}
        ]}
        a = build_tool_messages(record, trace, False, 500)
        b = build_tool_messages(record, trace, True, 500)
        instr_a, ia = to_responses_input(a)
        instr_b, ib = to_responses_input(b)
        self.assertEqual(instr_a, instr_b)
        self.assertEqual(len(ia), 5)
        self.assertEqual([v['type'] for v in ia[1:]], [
            'function_call', 'function_call_output',
            'function_call', 'function_call_output'])
        self.assertEqual(ia[2], ib[2])
        self.assertEqual(ia[4], ib[4])
        self.assertEqual(ia[1]['call_id'], ia[2]['call_id'])
        self.assertEqual(ia[3]['call_id'], ia[4]['call_id'])
        self.assertEqual(__import__('json').loads(ia[1]['arguments'])['query'], '')
        self.assertEqual(__import__('json').loads(ib[1]['arguments'])['query'], 'Book A author')
        self.assertEqual(__import__('json').loads(ib[3]['arguments'])['query'], 'writer hometown')

    def test_api_requests_correct_effort_and_usage(self):
        from types import SimpleNamespace
        from experiment import Luna

        class FakeResponses:
            def __init__(self):
                self.calls = []
            def create(self, **kw):
                self.calls.append(kw)
                return SimpleNamespace(
                    output_text='FINAL_ANSWER: Paris',
                    usage=SimpleNamespace(input_tokens=100, output_tokens=35),
                    status='completed', incomplete_details=None,
                )
        api = FakeResponses()
        llm = Luna(None, 'gpt-6-luna', 30, client=SimpleNamespace(responses=api))
        one = llm.ask('system', 'question', max_tokens=128, thinking=False)
        two = llm.ask('system', 'question', max_tokens=3072, thinking=True)
        self.assertEqual(one['prompt_tokens'], 100)
        self.assertEqual(two['completion_tokens'], 35)
        self.assertEqual(api.calls[0]['reasoning'], {'effort': 'none'})
        self.assertEqual(api.calls[1]['reasoning'], {'effort': 'medium'})
        self.assertEqual(api.calls[1]['model'], 'gpt-6-luna')
        self.assertEqual(api.calls[1]['store'], False)
        self.assertNotIn('temperature', api.calls[1])

if __name__ == '__main__':
    unittest.main()
