"""Optional compute backend helpers (NumPy / PyTorch).

This module keeps the project runnable without torch.
If torch is installed, impurity computations can run on torch backend.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import math


def _try_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def is_torch_available() -> bool:
    return _try_import_torch() is not None


def resolve_backend(preferred: str = "auto") -> str:
    p = str(preferred or "auto").strip().lower()
    if p not in {"auto", "numpy", "torch"}:
        return "numpy"
    if p == "torch":
        return "torch" if is_torch_available() else "numpy"
    if p == "auto":
        return "torch" if is_torch_available() else "numpy"
    return "numpy"


def resolve_torch_device(torch_device: str = "cpu") -> str:
    torch = _try_import_torch()
    if torch is None:
        return "cpu"

    d = str(torch_device or "cpu").strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return d or "cpu"


def _collect_labels_and_weights(
    instances: List,
    get_label: Callable,
    class_weights_dict: Optional[Dict],
) -> Tuple[List[int], Optional[List[float]]]:
    labels: List[int] = []
    weights: Optional[List[float]] = [] if class_weights_dict else None

    for inst in instances:
        label = get_label(inst)
        if label is None:
            continue
        try:
            y = int(label)
        except Exception:
            continue
        labels.append(y)
        if weights is not None:
            weights.append(float(class_weights_dict.get(y, 1.0)))

    return labels, weights


def _bincount_numpy(labels: List[int], weights: Optional[List[float]] = None):
    if not labels:
        return []
    max_label = max(labels)
    size = max(2, max_label + 1)
    counts = [0.0] * size
    if weights is None:
        for y in labels:
            counts[y] += 1.0
    else:
        for y, w in zip(labels, weights):
            counts[y] += float(w)
    return counts


def _bincount_torch(labels: List[int], weights: Optional[List[float]], device: str):
    torch = _try_import_torch()
    if torch is None or not labels:
        return None

    y = torch.tensor(labels, dtype=torch.long, device=device)
    if weights is None:
        c = torch.bincount(y, minlength=max(2, int(torch.max(y).item()) + 1))
    else:
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        c = torch.bincount(y, weights=w, minlength=max(2, int(torch.max(y).item()) + 1))
    return c


def calculate_gini(
    instances: List,
    get_label: Callable,
    class_weights_dict: Optional[Dict] = None,
    backend: str = "numpy",
    torch_device: str = "cpu",
) -> float:
    labels, weights = _collect_labels_and_weights(instances, get_label, class_weights_dict)
    if not labels:
        return 0.0

    backend = resolve_backend(backend)

    if backend == "torch":
        torch = _try_import_torch()
        if torch is not None:
            dev = resolve_torch_device(torch_device)
            c = _bincount_torch(labels, weights, dev)
            if c is not None:
                total = torch.sum(c)
                if float(total.item()) <= 0.0:
                    return 0.0
                probs = c[c > 0] / total
                g = 1.0 - torch.sum(probs * probs)
                return float(g.item())

    counts = _bincount_numpy(labels, weights)
    total = sum(counts)
    if total <= 0:
        return 0.0
    gini = 1.0
    for count in counts:
        if count > 0:
            p = count / total
            gini -= p * p
    return float(gini)


def calculate_entropy(
    instances: List,
    get_label: Callable,
    class_weights_dict: Optional[Dict] = None,
    backend: str = "numpy",
    torch_device: str = "cpu",
) -> float:
    labels, weights = _collect_labels_and_weights(instances, get_label, class_weights_dict)
    if not labels:
        return 0.0

    backend = resolve_backend(backend)

    if backend == "torch":
        torch = _try_import_torch()
        if torch is not None:
            dev = resolve_torch_device(torch_device)
            c = _bincount_torch(labels, weights, dev)
            if c is not None:
                total = torch.sum(c)
                if float(total.item()) <= 0.0:
                    return 0.0
                probs = c[c > 0] / total
                e = -torch.sum(probs * torch.log2(probs))
                return float(e.item())

    counts = _bincount_numpy(labels, weights)
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return float(entropy)
