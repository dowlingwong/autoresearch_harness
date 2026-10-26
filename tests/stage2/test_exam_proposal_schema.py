"""Opt-in exam schema transport and failure semantics. HTTP fixtures only; no inference."""
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from autoresearch.llm.providers import manager_response_format
from autoresearch.manager.base import ManagerStatus
from autoresearch.manager.langgraph_manager import LangGraphManager
from autoresearch.manager.random_valid import RandomValidManager
from autoresearch.memory.summarizer import MemoryMode, build_memory_context
from autoresearch.nodes.registry import load_registered_node
from autoresearch.nodes.active_keys import resolve_active_keys

ROOT=Path(__file__).resolve().parents[2]
GOOD={'rationale':'r','summary':'s','objective':'Change MAX_ITER from 200 to 201.',
      'param':'MAX_ITER','old_value':'200','new_value':'201'}


@contextmanager
def server(content):
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));requests.append(body)
            payload={'id':'fixture','object':'chat.completion','created':1,'model':'fixture',
                     'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':content}}],
                     'usage':{'prompt_tokens':7,'completion_tokens':5,'total_tokens':12,'completion_tokens_details':{'reasoning_tokens':2}}}
            raw=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler);t=threading.Thread(target=http.serve_forever,daemon=True);t.start()
    try:
        with patch.dict('os.environ',{'LLAMACPP_BASE_URL':f'http://127.0.0.1:{http.server_address[1]}/v1'}): yield requests
    finally: http.shutdown();http.server_close();t.join()


def propose(schema):
    spec=load_registered_node('digits_mlp',repo_root=ROOT)
    status=ManagerStatus('digits_mlp__s9001',2,.5,spec.metric_name,spec.metric_direction,{'MAX_ITER':'200'})
    return LangGraphManager(model='llamacpp/fixture',proposal_schema=schema,transport_retry=None).propose_next_trial(
        status,build_memory_context([],MemoryMode.NONE,spec,2),spec)


def test_schema_request_is_structure_only_rationale_first_and_records_usage():
    with server('<think>private reasoning fixture</think>'+json.dumps(GOOD)) as calls:
        p=propose('manager_proposal/v1')
    fmt=calls[0]['response_format'];schema=fmt['json_schema']['schema']
    assert fmt['type']=='json_schema'
    assert list(schema['properties'])==['rationale','summary','objective','param','old_value','new_value']
    assert all(v=={'type':'string'} for v in schema['properties'].values())
    assert 'enum' not in json.dumps(schema)
    assert p.extra['llm_call']['schema_version']=='manager_proposal/v1'
    assert p.extra['llm_call']['reasoning_tokens']==2
    assert not p.extra['parse_failed'] and '<think>' not in p.objective


def test_malformed_schema_reply_is_recorded_never_heuristic_replaced():
    for content in ['garbage',json.dumps({**GOOD,'new_value':2}),json.dumps({'summary':'s'})]:
        with server(content): p=propose('manager_proposal/v1')
        assert p.objective=='' and p.extra['parse_failed']
        assert p.extra['llm_call']['parse_failed'] and not p.extra['llm_call']['parse_ok']
        assert 'structured_edit' not in p.extra


def test_free_generation_keeps_request_and_historical_fallback():
    with server('garbage') as calls: p=propose(None)
    assert 'response_format' not in calls[0]
    assert 'schema_version' not in p.extra['llm_call']
    assert p.objective.startswith('Propose exactly one bounded change')


def test_schema_does_not_validate_or_enumerate_parameter_values():
    with server(json.dumps({**GOOD,'param':'invented_parameter','new_value':'invented_value'})):
        p=propose('manager_proposal/v1')
    assert not p.extra['parse_failed']


def test_random_valid_is_deterministic_and_never_chooses_inactive_keys():
    spec=load_registered_node('openml_credit_approval',repo_root=ROOT)
    constants={k:'1' for k in spec.editable_symbols};constants['model_type']='logistic_regression'
    active=resolve_active_keys(spec,constants);assert active
    grid={k:['1','2'] for k in spec.editable_symbols};grid['model_type']=['logistic_regression']
    a=RandomValidManager(grid);b=RandomValidManager(grid);seen=set()
    for trial in range(1,201):
        st=ManagerStatus('openml_credit_approval__s9001',trial,.5,spec.metric_name,spec.metric_direction,constants)
        p=a.propose_next_trial(st,None,spec);q=b.propose_next_trial(st,None,spec)
        assert p.to_dict()==q.to_dict()
        edit=p.extra['structured_edit'];assert edit['symbol'] in active
        assert edit['new'] in grid[edit['symbol']];seen.add(edit['symbol'])
    assert seen==set(active)


def test_empty_but_structural_strings_are_not_replaced_by_heuristics():
    with server(json.dumps({k:'' for k in GOOD})): p=propose('manager_proposal/v1')
    assert p.objective=='' and p.proposal_summary==''
    assert not p.extra['parse_failed'] and p.extra['proposal_precondition_failed']
