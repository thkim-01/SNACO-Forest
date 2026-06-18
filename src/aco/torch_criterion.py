import torch
from typing import Tuple, Dict, Optional, Any
import numpy as np

def init_torch_device(gpu_id: int = 0) -> torch.device:
    """Initialize and return the appropriate PyTorch device."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def _torch_entropy(y: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculate Shannon Entropy using PyTorch operations."""
    if len(y) == 0:
        return torch.tensor(0.0, device=y.device)
    
    if weights is not None:
        # Use simple bincount if y consists of 0s and 1s, but we do weighted sum
        counts = torch.bincount(y.long(), weights=weights[y.long()])
    else:
        counts = torch.bincount(y.long())
        
    probs = counts.float() / (counts.sum() + 1e-12)
    probs = probs[probs > 0]
    return -torch.sum(probs * torch.log2(probs))

def _torch_gini(y: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculate Gini Impurity using PyTorch operations."""
    if len(y) == 0:
        return torch.tensor(0.0, device=y.device)
    
    if weights is not None:
        counts = torch.bincount(y.long(), weights=weights[y.long()])
    else:
        counts = torch.bincount(y.long())
        
    probs = counts.float() / (counts.sum() + 1e-12)
    return 1.0 - torch.sum(probs ** 2)

def torch_find_best_threshold(
    x_col: torch.Tensor,
    y: torch.Tensor,
    criterion: str = "entropy",
    n_candidates: int = 32,
    torch_weights: Optional[torch.Tensor] = None
) -> Tuple[float, float, str]:
    """
    GPU-accelerated function to find the best split threshold.
    
    Args:
        x_col: 1D PyTorch tensor representing feature values for the active subset.
        y: 1D PyTorch tensor representing labels for the active subset.
        criterion: Impurity reduction criteria ("entropy" or "gini").
        n_candidates: Number of candidate thresholds to evaluate.
        torch_weights: 1D PyTorch tensor for class weights.
        
    Returns:
        Tuple of (best_threshold, best_info_gain, operator)
    """
    if len(x_col) < 2:
        return float(torch.median(x_col).item()), 0.0, "<="
        
    if len(x_col) > 5000:
        # Mini-batch logic on GPU for massive arrays
        idx = torch.randperm(len(x_col), device=x_col.device)[:5000]
        x_col = x_col[idx]
        y = y[idx]
        
    n_candidates = min(64, max(16, len(x_col) // 20))
    
    # Calculate quantiles for thresholds on GPU
    # torch.quantile requires float32
    x_float = x_col.float()
    q = torch.linspace(0.05, 0.95, n_candidates, device=x_col.device)
    thresholds = torch.unique(torch.quantile(x_float, q))
    
    impurity_fn = _torch_entropy if criterion == "entropy" else _torch_gini
    
    parent_impurity = impurity_fn(y, torch_weights)
    total_weight = len(y) if torch_weights is None else torch_weights[y.long()].sum()
    
    best_ig = -1.0
    best_t = float(torch.median(x_col).item())
    
    for t in thresholds:
        left_mask = x_col <= t
        right_mask = ~left_mask
        
        n_left = left_mask.sum()
        n_right = right_mask.sum()
        
        if n_left == 0 or n_right == 0:
            continue
            
        y_left = y[left_mask]
        y_right = y[right_mask]
        
        w_left = n_left.float() if torch_weights is None else torch_weights[y_left.long()].sum()
        w_right = n_right.float() if torch_weights is None else torch_weights[y_right.long()].sum()
        
        h_left = impurity_fn(y_left, torch_weights)
        h_right = impurity_fn(y_right, torch_weights)
        
        # IG = H_parent - (W_L/W_total * H_L + W_R/W_total * H_R)
        ig = parent_impurity - ((w_left / total_weight) * h_left + (w_right / total_weight) * h_right)
        
        ig_val = ig.item()
        if ig_val > best_ig:
            best_ig = ig_val
            best_t = t.item()
            
    # Fallback bounds check
    if best_ig < 0:
        best_ig = 0.0
        
    return best_t, best_ig, "<="
