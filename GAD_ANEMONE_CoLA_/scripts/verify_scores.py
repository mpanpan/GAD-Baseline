#!/usr/bin/env python3
"""Independently recompute metrics from persisted full-graph scores and graph masks."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad_handoff.data import load_graph,labels_and_masks,file_sha256
from gad_handoff.metrics import evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--config',default='configs/experiment.json')
    p.add_argument('--results',default='results')
    p.add_argument('--atol',type=float,default=1e-9)
    a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    failures=[];seen=0
    for ds in cfg['datasets']:
        graph,path=load_graph(a.data_root, ds)
        sha=file_sha256(path)
        n=graph.num_nodes()
        for seed in cfg['seeds']:
            labels,va,te=labels_and_masks(graph,seed)
            for model in cfg['models']:
                rd=Path(a.results)/model/ds/f'seed_{seed}'
                try:
                    doc=json.loads((rd/'metrics.json').read_text(encoding='utf-8'))
                    if doc['status']!='SUCCESS' or doc['dataset_sha256']!=sha or doc['seed']!=seed:
                        raise ValueError('Invalid provenance')
                    with np.load(rd/'scores.npz', allow_pickle=False) as saved:
                        scores=saved['scores'].copy()
                        ids=saved['node_ids'].copy()
                    expected_ids=(graph.ndata['source_tx_id'].detach().cpu().numpy()
                        if 'source_tx_id' in graph.ndata else np.arange(n))
                    if not np.array_equal(ids,expected_ids):
                        raise ValueError('Persisted node IDs not in exact graph order')
                    actual=evaluate(labels,scores,va,te,node_ids=ids)
                    for key,value in actual.items():
                        reported=doc['metrics'].get(key)
                        if not isinstance(reported,(int,float)) or not math.isclose(float(value),float(reported),rel_tol=0,abs_tol=a.atol):
                            raise ValueError(f'metric mismatch: {key}: {value} vs {reported}')
                    seen+=1
                except Exception as e:
                    failures.append((model,ds,seed,str(e)))
        del graph
    print(f'Independent score audit validated {seen}/80 files; failures={len(failures)}')
    for item in failures:print('INVALID',*item)
    if failures:raise SystemExit(2)
if __name__=='__main__':main()
