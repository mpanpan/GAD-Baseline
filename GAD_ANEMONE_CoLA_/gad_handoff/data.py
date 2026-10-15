"""Load the four files published in mpanpan/GAD-datasets/dataset."""
from pathlib import Path
import csv
import hashlib
import os
import numpy as np

NAMES = {"amazon", "yelp", "elliptic", "tfinance"}
EXPECTED_NODES = {"amazon": 11944, "yelp": 45954, "elliptic": 203769, "tfinance": 39357}
ELLIPTIC_FILES = ("elliptic_txs_classes.csv", "elliptic_txs_features.csv", "elliptic_txs_edgelist.csv")


def dataset_path(data_root, name):
    if name not in NAMES:
        raise ValueError(f"Unsupported dataset: {name!r}")
    base = Path(os.path.abspath(os.path.expanduser(str(data_root))))
    if not base.is_dir() or base.is_symlink():
        raise ValueError(f"Data root must be an accessible real directory: {base}")
    if "hetero" in {part.lower() for part in base.parts}:
        raise ValueError(f"Data root must not be inside hetero/: {base}")
    if name == "elliptic":
        path = base / "elliptic_bitcoin"
        if not path.is_dir() or path.is_symlink() or path.parent != base:
            raise ValueError(f"Expected direct elliptic_bitcoin directory under {base}")
        missing = [filename for filename in ELLIPTIC_FILES if not (path / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"Elliptic CSV files missing: {missing}")
        return path
    matches = [path for path in (base / name, base / (name + ".bin")) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one direct {name} or {name}.bin file under {base}; found {matches}")
    path = matches[0]
    if path.is_symlink() or path.parent != base or "hetero" in {part.lower() for part in path.parts}:
        raise ValueError(f"Unacceptable dataset path: {path}")
    return path


def file_sha256(path, chunk=8 * 1024 * 1024):
    """Hash file content, or all three Elliptic CSVs in fixed named order."""
    path = Path(path)
    h = hashlib.sha256()
    files = [path / name for name in ELLIPTIC_FILES] if path.is_dir() else [path]
    for file in files:
        if path.is_dir():
            h.update(file.name.encode("utf-8") + b"\0")
        with file.open("rb") as stream:
            while part := stream.read(chunk):
                h.update(part)
    return h.hexdigest()


def _get_1d(graph, key):
    if key not in graph.ndata:
        raise KeyError(f"Missing {key!r}; found {list(graph.ndata.keys())}")
    value = graph.ndata[key].detach().cpu().numpy()
    if value.ndim > 1 and value.shape[1] == 1:
        value = value.reshape(-1)
    if value.ndim != 1 or len(value) != graph.num_nodes():
        raise ValueError(f"{key} invalid shape {value.shape}")
    return value


def labels_and_masks(graph, trial):
    if trial not in range(10):
        raise ValueError("trial must be 0..9")
    y = _get_1d(graph, "label")
    if not np.isin(y, [-1, 0, 1]).all():
        raise ValueError(f"Unverified label coding: {np.unique(y)}")
    masks = {}
    for key in ("val_masks", "test_masks"):
        if key not in graph.ndata:
            raise KeyError(f"Missing trial masks: {key}")
        mask = graph.ndata[key].detach().cpu().numpy()
        if mask.ndim != 2 or mask.shape[0] != graph.num_nodes() or mask.shape[1] < 10:
            raise ValueError(f"{key} is not an Nx>=10 mask matrix: {mask.shape}")
        if not np.isin(mask[:, trial], [0, 1]).all():
            raise ValueError(f"{key} is not boolean")
        masks[key] = mask[:, trial].astype(bool) & (y >= 0)
    va, te = masks["val_masks"], masks["test_masks"]
    if np.any(va & te):
        raise ValueError("Known-label validation/test masks overlap")
    for name, mask in (("validation", va), ("test", te)):
        if set(np.unique(y[mask])) != {0, 1}:
            raise ValueError(f"{name} known-label set lacks both classes")
    return y.astype(np.int64), va, te


def _elliptic_masks(labels):
    """Ten deterministic, class-stratified 10/10/80 known-label splits."""
    n = len(labels)
    train = np.zeros((n, 10), dtype=bool)
    val = np.zeros((n, 10), dtype=bool)
    test = np.zeros((n, 10), dtype=bool)
    for trial in range(10):
        rng = np.random.default_rng(trial)
        for cls in (0, 1):
            ids = rng.permutation(np.flatnonzero(labels == cls))
            n_train = round(len(ids) * 0.1)
            n_val = round(len(ids) * 0.1)
            train[ids[:n_train], trial] = True
            val[ids[n_train:n_train + n_val], trial] = True
            test[ids[n_train + n_val:], trial] = True
    return train, val, test


def _load_elliptic(path):
    try:
        import dgl
        import torch
    except ImportError as exc:
        raise RuntimeError("DGL and PyTorch are required to load Elliptic") from exc
    classes, features, edges = (path / filename for filename in ELLIPTIC_FILES)
    with classes.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["txId", "class"]:
            raise ValueError("Unexpected Elliptic class CSV header")
        codes = {}
        for row in reader:
            txid = row["txId"]
            if not txid.isdecimal() or txid in codes or row["class"] not in {"1", "2", "unknown"}:
                raise ValueError("Invalid, duplicate or unmapped Elliptic class row")
            codes[txid] = {"1": 1, "2": 0, "unknown": -1}[row["class"]]
    ids = sorted(codes, key=int)
    if len(ids) != EXPECTED_NODES["elliptic"]:
        raise ValueError(f"Unexpected Elliptic node count: {len(ids)}")
    lookup = {txid: i for i, txid in enumerate(ids)}
    labels = np.asarray([codes[txid] for txid in ids], dtype=np.int64)
    seen = np.zeros(len(ids), dtype=bool)
    matrix = None
    with features.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) < 3 or row[0] not in lookup:
                raise ValueError("Unexpected Elliptic feature row")
            if matrix is None:
                matrix = np.empty((len(ids), len(row) - 2), dtype=np.float32)
            if len(row) != matrix.shape[1] + 2:
                raise ValueError("Inconsistent Elliptic feature width")
            index = lookup[row[0]]
            if seen[index]:
                raise ValueError("Duplicate Elliptic feature txId")
            values = np.asarray(row[2:], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite Elliptic features")
            matrix[index] = values
            seen[index] = True
    if matrix is None or not seen.all():
        raise ValueError("Missing Elliptic feature txIds")
    source, target = [], []
    with edges.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["txId1", "txId2"]:
            raise ValueError("Unexpected Elliptic edge CSV header")
        for row in reader:
            if row["txId1"] not in lookup or row["txId2"] not in lookup:
                raise ValueError("Unknown Elliptic edge endpoint")
            source.append(lookup[row["txId1"]])
            target.append(lookup[row["txId2"]])
    graph = dgl.graph((torch.tensor(source), torch.tensor(target)), num_nodes=len(ids))
    graph.ndata["feat"] = torch.from_numpy(matrix)
    graph.ndata["label"] = torch.from_numpy(labels)
    graph.ndata["source_tx_id"] = torch.tensor([int(txid) for txid in ids], dtype=torch.int64)
    train, val, test = _elliptic_masks(labels)
    graph.ndata["train_masks"] = torch.from_numpy(train)
    graph.ndata["val_masks"] = torch.from_numpy(val)
    graph.ndata["test_masks"] = torch.from_numpy(test)
    return graph


def load_graph(data_root, name):
    path = dataset_path(data_root, name)
    if name == "elliptic":
        graph = _load_elliptic(path)
    else:
        try:
            from dgl.data.utils import load_graphs
        except ImportError as exc:
            raise RuntimeError("DGL is required to load published binary datasets") from exc
        graphs, _ = load_graphs(str(path))
        if len(graphs) != 1:
            raise ValueError(f"Expected one DGL graph, found {len(graphs)}")
        graph = graphs[0]
    if not graph.is_homogeneous:
        raise ValueError("Heterogeneous graph encountered")
    if graph.num_nodes() != EXPECTED_NODES[name]:
        raise ValueError(f"Unexpected node count {graph.num_nodes()} for {name}")
    if "feat" not in graph.ndata or graph.ndata["feat"].shape[0] != graph.num_nodes():
        raise ValueError("Missing or misaligned node features")
    if not graph.ndata["feat"].isfinite().all():
        raise ValueError("Nonfinite node features")
    return graph, path
