#!/usr/bin/env python3
"""Report only fully verified ten-run groups; never impute missing runs."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

METRIC_KEYS = [
    'AUROC', 'AUPRC_AP', 'Recall@K_oracle', 'Precision', 'Recall', 'F1',
    'Macro-F1', 'G-Mean', 'Precision@K_oracle', 'Balanced_Accuracy',
    'Specificity', 'MCC', 'Accuracy', 'Precision@1pct', 'Recall@1pct',
    'Precision@5pct', 'Recall@5pct', 'PR_AUC_trapezoid',
]
RESOURCE_KEYS = [
    'train_seconds', 'inference_seconds', 'total_elapsed_seconds',
    'peak_rss_bytes', 'peak_rss_delta_bytes', 'peak_cuda_allocated_bytes',
]


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.json')
    parser.add_argument('--results', default='results')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    expected_runs = len(config['models']) * len(config['datasets']) * len(config['seeds'])
    root = Path(args.results)
    root.mkdir(parents=True, exist_ok=True)
    per_run, summary, missing = [], [], []
    for model in config['models']:
        for dataset in config['datasets']:
            group = []
            for seed in config['seeds']:
                run_dir = root / model / dataset / f'seed_{seed}'
                try:
                    record = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
                    if (record.get('status'), record.get('model'), record.get('dataset'),
                        record.get('seed'), record.get('trial')) != (
                            'SUCCESS', model, dataset, seed, seed):
                        raise ValueError('inconsistent run identity/status')
                    if not (run_dir / 'scores.npz').is_file():
                        raise ValueError('scores.npz missing')
                    values = {key: record['metrics'][key] for key in METRIC_KEYS}
                    values.update({key: record[key] for key in RESOURCE_KEYS})
                    if any(not isinstance(value, (int, float)) or not math.isfinite(value)
                           for value in values.values()):
                        raise ValueError('missing or nonfinite metric/resource')
                    if any(value < 0 for value in (record[key] for key in RESOURCE_KEYS)):
                        raise ValueError('negative time or memory')
                    group.append(record)
                    method_metadata = record.get('method_metadata', {})
                    per_run.append({'model': model, 'dataset': dataset, 'seed': seed,
                                    **values, 'source_git_sha': record['source_git_sha'],
                                    'dataset_sha256': record['dataset_sha256'],
                                    'score_method': method_metadata.get('score_method', 'UNRECORDED'),
                                    'label_use': method_metadata.get('label_use', 'UNRECORDED')})
                except Exception as exc:
                    missing.append({'model': model, 'dataset': dataset, 'seed': seed,
                                    'reason': str(exc)})
            row = {'model': model, 'dataset': dataset, 'completed': len(group),
                   'expected': len(config['seeds']),
                   'status': 'COMPLETE' if len(group) == len(config['seeds']) else 'INCOMPLETE'}
            if len(group) == len(config['seeds']):
                sources = {r['source_git_sha'] for r in group}
                datasets = {r['dataset_sha256'] for r in group}
                protocols = {(r.get('method_metadata', {}).get('score_method'),
                              r.get('method_metadata', {}).get('label_use')) for r in group}
                if len(sources) != 1 or 'UNAVAILABLE' in sources or len(datasets) != 1 or len(protocols) != 1:
                    row['status'] = 'INVALID_PROVENANCE'
                else:
                    row['score_method'], row['label_use'] = next(iter(protocols))
                    if not row['score_method'] or not row['label_use']:
                        row['status'] = 'INVALID_SCORE_PROTOCOL'
                if row['status'] == 'COMPLETE':
                    for key in METRIC_KEYS:
                        numbers = [r['metrics'][key] for r in group]
                        row[key + '_mean'] = statistics.mean(numbers)
                        row[key + '_std'] = statistics.stdev(numbers)
                    for key in RESOURCE_KEYS:
                        numbers = [r[key] for r in group]
                        row[key + '_mean'] = statistics.mean(numbers)
                        row[key + '_std'] = statistics.stdev(numbers)
                        if key.endswith('_bytes'):
                            row[key + '_max'] = max(numbers)
                        else:
                            row[key + '_sum'] = sum(numbers)
            summary.append(row)
    save_csv(root / 'per_run.csv', per_run)
    save_csv(root / 'summary.csv', summary)
    save_csv(root / 'missing.csv', missing)
    header = ['Method', 'Dataset', 'Completed', 'Score method', 'Label use', *METRIC_KEYS,
              'Training Time (s)', 'Inference Time (s)', 'Peak RSS (MiB)', 'Peak CUDA (MiB)']
    lines = [
        '# ANEMONE / CoLA / GREET on GAD-datasets', '',
        'All values are mean ± sample SD over ten independent runs. AP is reported as AUPRC.',
        'Classification thresholds use validation labels. Recall@K and Precision@K use oracle K = test positives.',
        'Elliptic uses deterministic class-stratified 10/10/80 splits; other files use their published masks.',
        '', '| ' + ' | '.join(header) + ' |',
        '|' + '|'.join('---' for _ in header) + '|',
    ]
    for row in summary:
        cells = [row['model'], row['dataset'], f"{row['completed']}/{row['expected']}",
                 str(row.get('score_method', 'UNRESOLVED')),
                 str(row.get('label_use', 'UNRESOLVED'))]
        for key in [*METRIC_KEYS, 'train_seconds', 'inference_seconds',
                    'peak_rss_bytes', 'peak_cuda_allocated_bytes']:
            if row['status'] != 'COMPLETE':
                cells.append('INCOMPLETE')
            else:
                scale = 1048576 if key.endswith('_bytes') else 1
                cells.append(f"{row[key + '_mean']/scale:.4f} ± {row[key + '_std']/scale:.4f}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', f'Validated runs: {len(per_run)}/{expected_runs}; missing or invalid: {len(missing)}.',
              'Time totals and memory maxima are separate; peak memory is never summed.',
              'Per-run resources and ten-run statistics are in `per_run.csv` and `summary.csv`.']
    (root / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Complete {len(per_run)}/{expected_runs}; missing {len(missing)}; outputs in {root}')
    if missing or any(row['status'] != 'COMPLETE' for row in summary):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
