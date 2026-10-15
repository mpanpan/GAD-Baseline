"""Execute one strictly defined model/dataset/seed after a port is implemented."""
import argparse
import importlib
import json
import os
import platform
import subprocess
import time
import traceback
import threading
from pathlib import Path
import numpy as np
from gad_handoff.data import load_graph, labels_and_masks, file_sha256
from gad_handoff.metrics import evaluate


def git_rev(path):
    try:
        return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return 'UNAVAILABLE'


def run(args):
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.model not in cfg['models'] or args.dataset not in cfg['datasets'] or args.seed not in cfg['seeds']:
        raise ValueError('Unapproved method / dataset / seed')
    if cfg['model_config'][args.model]['status'] != 'APPROVED':
        raise RuntimeError(f"Model config {args.model} NOT APPROVED. Agent must complete port, parity and hyperparameter approval, then change status to APPROVED")
    if args.model == 'greet':
        method = cfg['model_config']['greet'].get('score_method')
        label_use = cfg['model_config']['greet'].get('label_use')
        if method in (None, 'UNDECIDED') or label_use in (None, 'UNDECIDED'):
            raise RuntimeError('GREET anomaly scoring and label-use protocol are unresolved — BLOCKED')
    run_dir = Path(args.output_root)/args.model/args.dataset/f'seed_{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        adapter = importlib.import_module(f'adapters.{args.model}')
    except ImportError as exc:
        raise RuntimeError(f'Original model adapter adapters/{args.model}.py is NOT yet implemented — BLOCKED') from exc
    if not hasattr(adapter, 'train_and_score'):
        raise RuntimeError('Adapter missing train_and_score')
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    t0 = time.monotonic()
    graph, graph_file = load_graph(args.data_root, args.dataset)
    y, va, te = labels_and_masks(graph, args.seed)
    # Adapter must not read graph.ndata['label'] or split masks; enforce with a *features-only* graph.
    # Keep edge topology; a shallow clone of graph does not mutate original ndata mapping.
    train_graph = graph.local_var()
    for k in list(train_graph.ndata.keys()):
        if k != 'feat':
            del train_graph.ndata[k]
    # Preserve edge attributes if the author method requires them; never expose labels or split masks.
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError('psutil is required for per-run memory measurement') from exc
    process = psutil.Process()
    rss_start = process.memory_info().rss
    peak_rss = [rss_start]
    stop = threading.Event()
    def sample_memory():
        while not stop.wait(0.05):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    monitor = threading.Thread(target=sample_memory, daemon=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    monitor.start()
    try:
        result = adapter.train_and_score(train_graph, seed=args.seed, device=args.device,
            config=cfg['model_config'][args.model], run_dir=str(run_dir))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        stop.set()
        monitor.join()
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
    peak_cuda_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    if not isinstance(result, dict) or 'scores' not in result:
        raise TypeError('train_and_score must return a dict with all-node scores')
    scores = np.asarray(result['scores'], dtype=np.float64)
    if scores.shape != (graph.num_nodes(),):
        raise ValueError(f'Adapter score shape {scores.shape}, expected {(graph.num_nodes(),)}')
    node_ids = (graph.ndata['source_tx_id'].detach().cpu().numpy()
        if 'source_tx_id' in graph.ndata else np.arange(graph.num_nodes(), dtype=np.int64))
    metrics = evaluate(y, scores, va, te, node_ids=node_ids)
    metadata = result.get('metadata', {})
    if not isinstance(metadata, dict):
        raise TypeError('Adapter metadata must be object')
    for key in ('score_method', 'label_use'):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f'Adapter must disclose {key}')
    if args.model == 'greet':
        for key in ('score_method', 'label_use'):
            if metadata.get(key) != cfg['model_config']['greet'][key]:
                raise ValueError(f'GREET {key} differs from approved configuration — BLOCKED')
        if metadata['label_use'] != 'none':
            raise RuntimeError('Supervised GREET probe needs an explicit evaluator-controlled label-fit stage — BLOCKED')
    for t in ('train_seconds','inference_seconds'):
        if t not in result or not np.isfinite(float(result[t])) or float(result[t]) < 0:
            raise ValueError(f'Invalid duration {t}')
    import torch
    repo = Path(__file__).resolve().parents[1]
    info = dict(status='SUCCESS', model=args.model, dataset=args.dataset, seed=args.seed,
        trial=args.seed, scores_direction='higher_is_anomaly',
        dataset_file=str(graph_file), dataset_sha256=file_sha256(graph_file),
        source_git_sha=git_rev(repo/'external'/{'cola':'CoLA','anemone':'ANEMONE','greet':'GREET'}[args.model]),
        handoff_git_sha=git_rev(repo), data_git_sha=git_rev(Path(os.path.abspath(args.data_root)).parent),
        dataset_source_git_sha=cfg['dataset_repo_commit'],
        config=cfg['model_config'][args.model], method_metadata=metadata,
        gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES','unset'), torch_version=torch.__version__,
        python_version=platform.python_version(), graph_nodes=int(graph.num_nodes()),
        graph_edges=int(graph.num_edges()), train_seconds=float(result['train_seconds']),
        inference_seconds=float(result['inference_seconds']),
        total_elapsed_seconds=0.0,
        peak_rss_bytes=int(peak_rss[0]), peak_rss_delta_bytes=int(peak_rss[0]-rss_start),
        peak_cuda_allocated_bytes=int(peak_cuda_allocated), metrics=metrics)
    tmp = run_dir/'scores.partial.npz'
    with open(tmp,'wb') as f:
        np.savez_compressed(f, node_ids=node_ids, scores=scores)
    os.replace(tmp, run_dir/'scores.npz')
    info['total_elapsed_seconds'] = float(time.monotonic()-t0)
    tmp_json = run_dir/'metrics.partial.json'
    tmp_json.write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp_json,run_dir/'metrics.json')
    print(json.dumps(dict(status='SUCCESS', model=args.model, dataset=args.dataset, seed=args.seed, metrics=metrics)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['cola','anemone','greet'],required=True)
    p.add_argument('--dataset', choices=['amazon','yelp','elliptic','tfinance'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output-root',default='results')
    p.add_argument('--config',default='configs/experiment.json')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    try:
        run(a)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

if __name__ == '__main__':
    main()
