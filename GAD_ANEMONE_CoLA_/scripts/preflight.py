#!/usr/bin/env python3
import argparse, json, platform, subprocess
from pathlib import Path
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad_handoff.data import load_graph, labels_and_masks, file_sha256, ELLIPTIC_FILES

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',default='results/data_audit.json')
    p.add_argument('--skip-sha256',action='store_true')
    a=p.parse_args()
    import torch
    import dgl
    info = {'python':platform.python_version(),'torch':torch.__version__,'dgl':dgl.__version__,
         'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(), 'datasets':{}}
    if torch.cuda.is_available():
        info['gpu_capabilities']=[str(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
        info['gpu_names']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for name in ['amazon','yelp','elliptic','tfinance']:
        graph, path=load_graph(a.data_root,name)
        rows={'path':str(path),'nodes':int(graph.num_nodes()),'edges':int(graph.num_edges()),
           'features_shape':list(graph.ndata['feat'].shape),'node_data_keys':list(graph.ndata.keys())}
        if not a.skip_sha256:
            rows['sha256']=file_sha256(path)
            if name == 'elliptic':
                rows['source_file_sha256']={filename:file_sha256(path/filename)
                                               for filename in ELLIPTIC_FILES}
        rows['split_source']=('deterministic_stratified_10_10_80'
                              if name == 'elliptic' else 'published_binary_trial_masks')
        label=graph.ndata['label'].detach().cpu().numpy().reshape(-1)
        rows['label_distribution']={str(k):int(v) for k,v in zip(*np.unique(label,return_counts=True))}
        rows['trials']={}
        for trial in range(10):
            y,v,t=labels_and_masks(graph, trial)
            rows['trials'][str(trial)]={'val_total':int(v.sum()),'test_total':int(t.sum()),
                  'val_anomalies':int(y[v].sum()),'test_anomalies':int(y[t].sum())}
        info['datasets'][name]=rows
        print(name,rows['nodes'],rows['edges'],rows['label_distribution'])
        del graph
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('PASS',out)
if __name__=='__main__':main()
