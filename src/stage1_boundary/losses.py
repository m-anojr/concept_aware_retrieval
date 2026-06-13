"""
Contrastive boundary loss for the Pedagogical Boundary Detector.

Two complementary terms are combined:

1. Adjacent-pair margin loss
   For every consecutive pair (t, t+1):
     - if they are in the SAME segment (no boundary), pull their
       embeddings together: minimize (1 - cos_sim)^2
     - if a boundary lies between them, push them apart with a margin:
       minimize max(0, cos_sim - (1 - margin))^2

2. Random in-segment / cross-segment InfoNCE term
   For each anchor time-step, sample one positive (another time-step in the
   same segment) and several negatives (time-steps in other segments), and
   apply an InfoNCE loss. This gives the encoder a non-local supervisory
   signal beyond immediate neighbours, which is especially useful for very
   short or noisy segments.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


def adjacent_margin_loss(z: torch.Tensor, is_boundary: torch.Tensor, margin: float,
                          pair_mask: torch.Tensor = None) -> torch.Tensor:
    """
    z: (B, T, E) L2-normalized embeddings
    is_boundary: (B, T-1) binary {0,1}
    pair_mask: optional (B, T-1) bool, True for valid (non-padded) pairs
    """
    sim = (z[:, :-1, :] * z[:, 1:, :]).sum(dim=-1)  # (B, T-1)

    same_mask = (is_boundary == 0).float()
    diff_mask = (is_boundary == 1).float()
    if pair_mask is not None:
        valid = pair_mask.float()
        same_mask = same_mask * valid
        diff_mask = diff_mask * valid

    pull_loss = same_mask * (1.0 - sim).pow(2)
    push_loss = diff_mask * F.relu(sim - (1.0 - margin)).pow(2)

    denom = same_mask.sum() + diff_mask.sum() + 1e-8
    return (pull_loss.sum() + push_loss.sum()) / denom


def sample_pairs(seg_ids: torch.Tensor, num_negatives: int,
                  valid_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each time-step (anchor), sample one positive index from the same
    segment and `num_negatives` indices from different segments.

    seg_ids: (B, T) integer segment ids per time-step
    valid_mask: optional (B, T) bool, True for non-padded time-steps. Only
                non-padded indices are considered as positive/negative
                candidates, and anchors at padded positions are marked
                invalid.

    Returns
    -------
    pos_idx: (B, T) long tensor of positive indices into the T dimension
    neg_idx: (B, T, num_negatives) long tensor of negative indices
    valid:   (B, T) bool mask - False where no valid positive exists
             (e.g. a segment of size 1, or the anchor itself is padding)
    """
    B, T = seg_ids.shape
    device = seg_ids.device
    pos_idx = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)
    neg_idx = torch.zeros(B, T, num_negatives, dtype=torch.long, device=device)
    valid = torch.zeros(B, T, dtype=torch.bool, device=device)

    seg_ids_np = seg_ids.detach().cpu().numpy()
    valid_np = valid_mask.detach().cpu().numpy() if valid_mask is not None else np.ones((B, T), dtype=bool)

    for b in range(B):
        ids = seg_ids_np[b]
        valid_t = valid_np[b]
        for t in range(T):
            if not valid_t[t]:
                continue

            same = ((ids == ids[t]) & valid_t).nonzero()[0]
            same = same[same != t]
            if len(same) > 0:
                pos_idx[b, t] = int(same[torch.randint(0, len(same), (1,)).item()])
                valid[b, t] = True

            diff = ((ids != ids[t]) & valid_t).nonzero()[0]
            if len(diff) > 0:
                choice = diff[torch.randint(0, len(diff), (num_negatives,)).numpy() % len(diff)]
                neg_idx[b, t] = torch.from_numpy(choice).to(device)
            else:
                valid[b, t] = False

    return pos_idx, neg_idx, valid


def segment_infonce_loss(z: torch.Tensor, seg_ids: torch.Tensor,
                          num_negatives: int = 4, temperature: float = 0.1,
                          valid_mask: torch.Tensor = None) -> torch.Tensor:
    """
    z: (B, T, E) L2-normalized embeddings
    seg_ids: (B, T) integer segment ids per time-step
    valid_mask: optional (B, T) bool, True for non-padded time-steps
    """
    B, T, E = z.shape
    pos_idx, neg_idx, valid = sample_pairs(seg_ids, num_negatives, valid_mask)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    batch_idx = torch.arange(B, device=z.device).view(B, 1).expand(B, T)

    z_anchor = z  # (B, T, E)
    z_pos = z[batch_idx, pos_idx]  # (B, T, E)
    z_neg = z[batch_idx.unsqueeze(-1).expand(-1, -1, num_negatives),
              neg_idx]  # (B, T, K, E)

    pos_sim = (z_anchor * z_pos).sum(dim=-1) / temperature  # (B, T)
    neg_sim = (z_anchor.unsqueeze(2) * z_neg).sum(dim=-1) / temperature  # (B, T, K)

    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, T, 1+K)
    labels = torch.zeros(B, T, dtype=torch.long, device=z.device)

    loss_per_step = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
    ).reshape(B, T)

    loss = (loss_per_step * valid.float()).sum() / (valid.float().sum() + 1e-8)
    return loss


def contrastive_boundary_loss(z: torch.Tensor, is_boundary: torch.Tensor, seg_ids: torch.Tensor,
                               margin: float = 0.5, num_negatives: int = 4,
                               temperature: float = 0.1, infonce_weight: float = 0.5,
                               key_padding_mask: torch.Tensor = None) -> torch.Tensor:
    """Combined Stage-1 training objective.

    key_padding_mask: optional (B, T) bool, True at PADDED positions
                       (the convention used by nn.Transformer). Internally
                       converted to "valid" masks for the loss terms.
    """
    valid_mask = None
    pair_mask = None
    if key_padding_mask is not None:
        valid_mask = ~key_padding_mask  # True where real data
        pair_mask = valid_mask[:, :-1] & valid_mask[:, 1:]

    l_margin = adjacent_margin_loss(z, is_boundary, margin, pair_mask=pair_mask)
    l_infonce = segment_infonce_loss(z, seg_ids, num_negatives, temperature, valid_mask=valid_mask)
    return l_margin + infonce_weight * l_infonce
