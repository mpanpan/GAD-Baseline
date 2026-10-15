#!/usr/bin/env python3
"""CPU orchestrator for independent H200 jobs; safe dry-run + success-only resume."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path


def parse_csv(x, converter=str):
    return [converter(v) for v in x.split(',') if v.strip()]


def valid_success(run_dir, model, dataset, seed):
    fp=run_dir/'metrics.json'
    sp=run_dir/'scores.npz'
    if not (fp.exists() and sp.exists()):
        return False
    try:
        doc=json.loads(fp.read_text(encoding='utf-8'))
        return (doc['status']=='SUCCESS' and doc['model']==model and
                doc['dataset']==dataset and doc['seed']==seed and
                'AUROC' in doc['metrics'] and 'AUPRC_AP' in doc['metrics'])
    except Exception:
        return False


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True)
    p.add_argument('--gpus',default='0')
    p.add_argument('--max-parallel',type=int,default=2)
    p.add_argument('--models',default='cola,anemone')
    p.add_argument('--datasets',default='amazon,yelp,elliptic,tfinance')
    p.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9')
    p.add_argument('--config',default='configs/experiment.json')
    p.add_argument('--output-root',default='results')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    gpus=parse_csv(a.gpus)
    models=parse_csv(a.models)
    datasets=parse_csv(a.datasets)
    seeds=parse_csv(a.seeds,int)
    if not gpus or len(gpus)!=len(set(gpus)) or a.max_parallel < 1:
        p.error('Unique GPU IDs and max-parallel>=1 required')
    if len(models)!=len(set(models)) or len(datasets)!=len(set(datasets)) or len(seeds)!=len(set(seeds)):
        p.error('Duplicate experiments requested')
    if not(set(models)<=set(cfg['models']) and set(datasets)<=set(cfg['datasets']) and set(seeds)<=set(cfg['seeds'])):
        p.error('Unapproved method/dataset/seed in configuration')
    jobs=[(model,dataset,seed) for model in models for dataset in datasets for seed in seeds]
    waiting=[]
    outroot=Path(a.output_root)
    for model,dataset,seed in jobs:
        rd=outroot/model/dataset/f'seed_{seed}'
        if valid_success(rd,model,dataset,seed):
            print('SKIP success',model,dataset,seed)
        else:
            waiting.append((model,dataset,seed))
    print(f'Approved matrix: {len(jobs)} tasks; pending {len(waiting)}; GPU slots={gpus}; cap={a.max_parallel}')
    if a.dry_run:
        for x in waiting:
            print('DRY-RUN',*x)
        return
    if any(cfg['model_config'][m]['status']!='APPROVED' for m in models):
        p.error('Adapters and hyperparameters not approved; see config status + AGENTS.md G2')
    outroot.mkdir(exist_ok=True,parents=True)
    running={}
    failed=[]
    free=list(gpus)
    while waiting or running:
        while waiting and free and len(running)<a.max_parallel:
            model,dataset,seed=waiting.pop(0)
            rd=outroot/model/dataset/f'seed_{seed}'
            rd.mkdir(parents=True,exist_ok=True)
            slot=free.pop(0)
            command=[sys.executable,'-m','gad_handoff.run_one','--model',model,
                '--dataset',dataset,'--seed',str(seed),'--data-root',a.data_root,
                '--output-root',a.output_root,'--config',a.config,'--device','cuda:0']
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=slot
            env['PYTHONUNBUFFERED']='1'
            if not a.retry_failed and (rd/'run.failed').exists():
                print('Previous failed run BLOCKED; use --retry-failed only after investigating:', rd)
                failed.append((model,dataset,seed,'existing failed run'))
                free.append(slot)
                continue
            (rd/'run.failed').unlink(missing_ok=True)
            f=open(rd/'run.log','a',encoding='utf-8')
            f.write('\nBEGIN '+json.dumps({'command':command,'physical_gpu':slot,'time':time.ctime()})+'\n');f.flush()
            process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env)
            running[process.pid]=(process,f,slot,model,dataset,seed,rd)
            print('START', model,dataset,seed,'GPU',slot,'PID',process.pid,flush=True)
        if not running:
            break
        time.sleep(1)
        for pid,record in list(running.items()):
            proc,f,slot,model,dataset,seed,rd=record
            rc=proc.poll()
            if rc is None:
                continue
            f.write('END '+json.dumps({'returncode':rc,'time':time.ctime()})+'\n');f.close()
            del running[pid];free.append(slot)
            if rc==0 and valid_success(rd,model,dataset,seed):
                print('SUCCESS',model,dataset,seed,flush=True)
            else:
                (rd/'run.failed').write_text(f'Return code {rc}; inspect run.log\n',encoding='utf-8')
                failed.append((model,dataset,seed,rc))
                print('FAILED',model,dataset,seed,'rc',rc,flush=True)
    if failed:
        print('Failures:',failed,file=sys.stderr)
        raise SystemExit(1)
    print('All requested tasks successful; run python scripts/report.py for strict 80/80 audit')

if __name__=='__main__':
    main()
