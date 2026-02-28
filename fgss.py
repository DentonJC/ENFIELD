from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

if TYPE_CHECKING:
    from ..templates import SupervisedTemplate


class FGSS_greedyPlugin(SupervisedPlugin):
    """GSS replay plugin (Greedy variant).

    Maintains an external memory buffer. Uses cosine similarity between
    feature-space gradients (dL/dh at penultimate features) to decide whether
    to replace memory items with current batch samples.
    """

    def __init__(self, mem_size: int = 200, mem_strength: int = 5, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = []
        self.mem_size = int(mem_size)
        self.mem_strength = int(mem_strength)

        # This device is used for model-related computations.
        self.device = torch.device("cpu")

        # Keep the buffer on CPU to save VRAM.
        self.ext_mem_list_x = torch.empty(self.mem_size, *input_size, dtype=torch.float32).fill_(0)
        self.ext_mem_list_y = torch.empty(self.mem_size, dtype=torch.long).fill_(0)
        self.ext_mem_list_current_index = 0

        # Store score per buffer slot (CPU).
        self.buffer_score = torch.empty(self.mem_size, dtype=torch.float32).fill_(0)

    def before_training(self, strategy: "SupervisedTemplate", **kwargs):
        # Keep self.device aligned with model device for computations.
        self.device = strategy.device

        # Ensure buffer stays on CPU.
        self.ext_mem_list_x = self.ext_mem_list_x.cpu()
        self.ext_mem_list_y = self.ext_mem_list_y.cpu()
        self.buffer_score = self.buffer_score.cpu()

    @staticmethod
    def cosine_similarity(x1: torch.Tensor, x2: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
        """Cosine similarity between rows of x1 and rows of x2."""
        x2 = x1 if x2 is None else x2
        if x1.dim() != 2 or x2.dim() != 2:
            raise ValueError(f"cosine_similarity expects 2D tensors, got {x1.shape=} {x2.shape=}")
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    @staticmethod
    def _get_classifier(model) -> torch.nn.Module:
        classifier = getattr(model, "classifier", None)
        if classifier is None:
            classifier = getattr(model, "fc", None)
        if classifier is None:
            classifier = getattr(model, "linear", None)
        if classifier is None:
            raise AttributeError(
                "Could not find a classifier layer on the model. Expected one of: "
                "model.classifier, model.fc, model.linear"
            )
        if not hasattr(classifier, "weight"):
            raise AttributeError("Classifier layer does not expose `.weight` as expected.")
        return classifier

    def _forward_with_penultimate(
        self, strategy: "SupervisedTemplate", x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs model forward and captures penultimate features h (input to classifier)."""
        model = strategy.model
        classifier = self._get_classifier(model)

        feats: Dict[str, torch.Tensor] = {}

        def hook(_module, inp, _out):
            feats["h"] = inp[0].detach()

        handle = classifier.register_forward_hook(hook)
        try:
            logits = model(x)
        finally:
            handle.remove()

        if "h" not in feats:
            raise RuntimeError("Failed to capture penultimate features via classifier forward hook.")
        return feats["h"], logits

    def _feature_grads_dldh(
        self, strategy: "SupervisedTemplate", x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Computes per-sample dL/dh in feature space (h = penultimate features)."""
        device = strategy.device
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        classifier = self._get_classifier(strategy.model)
        W = classifier.weight.to(device)  # [C, D]

        with torch.no_grad():
            h, logits = self._forward_with_penultimate(strategy, x)  # h: [B,D], logits: [B,C]
            probs = F.softmax(logits, dim=1)
            num_classes = probs.size(1)
            y_onehot = F.one_hot(y, num_classes=num_classes).float()
            grad_logits = probs - y_onehot  # [B,C]
            grad_h = grad_logits @ W        # [B,D]
        return grad_h

    def get_rand_mem_grads(self, strategy, _grad_dims, gss_batch_size: int) -> torch.Tensor:
        """Returns memory subset gradients in feature space: [num_mem_subs, hidden_dim]."""
        current_n = self.ext_mem_list_current_index
        if current_n == 0:
            return torch.zeros(0, device=strategy.device)

        temp_gss_batch_size = min(int(gss_batch_size), current_n)
        if temp_gss_batch_size == 0:
            return torch.zeros(0, device=strategy.device)

        # Use temp_gss_batch_size consistently.
        num_mem_subs = min(self.mem_strength, current_n // temp_gss_batch_size)
        if num_mem_subs == 0:
            return torch.zeros(0, device=strategy.device)

        device = strategy.device
        shuffled_inds = torch.randperm(current_n, device=torch.device("cpu"))

        mem_grads: Optional[torch.Tensor] = None
        for i in range(num_mem_subs):
            idx = shuffled_inds[i * temp_gss_batch_size : i * temp_gss_batch_size + temp_gss_batch_size]
            batch_x = self.ext_mem_list_x[idx].to(device)
            batch_y = self.ext_mem_list_y[idx].to(device)

            grad_h = self._feature_grads_dldh(strategy, batch_x, batch_y)  # [B,D]
            grad_vec = grad_h.mean(dim=0)  # [D]

            if mem_grads is None:
                hidden_dim = grad_vec.numel()
                mem_grads = torch.zeros(num_mem_subs, hidden_dim, dtype=torch.float32, device=device)

            mem_grads[i].copy_(grad_vec)

        return mem_grads if mem_grads is not None else torch.zeros(0, device=device)

    def get_batch_sim(self, strategy, grad_dims, batch_x: torch.Tensor, batch_y: torch.Tensor):
        """Max cosine similarity between current batch mean dL/dh and memory subset dL/dh."""
        device = strategy.device
        mem_grads = self.get_rand_mem_grads(strategy, grad_dims, gss_batch_size=len(batch_x))
        if mem_grads.numel() == 0:
            return torch.tensor(0.0, device=device), mem_grads

        grad_h = self._feature_grads_dldh(strategy, batch_x, batch_y)  # [B,D]
        batch_grad = grad_h.mean(dim=0, keepdim=True)                  # [1,D]

        sims = self.cosine_similarity(mem_grads, batch_grad)  # [N,1]
        batch_sim = sims.max()
        return batch_sim, mem_grads

    def get_each_batch_sample_sim(self, strategy, _grad_dims, mem_grads: torch.Tensor, batch_x, batch_y):
        """Cosine similarity per sample: max over memory subsets."""
        device = strategy.device
        batch_size = batch_x.size(0)

        if mem_grads.numel() == 0:
            return torch.zeros(batch_size, device=device)

        grad_h = self._feature_grads_dldh(strategy, batch_x, batch_y)  # [B,D]
        sims = self.cosine_similarity(mem_grads, grad_h)               # [N,B]
        cosine_sim, _ = sims.max(dim=0)                                # [B]
        return cosine_sim

    def before_training_exp(self, strategy, num_workers: int = 0, shuffle: bool = True, **kwargs):
        """Build dataloader mixing current experience data and replay memory."""
        if self.ext_mem_list_current_index == 0:
            return

        n = self.ext_mem_list_current_index
        mem_x = self.ext_mem_list_x[:n].cpu()
        mem_y = self.ext_mem_list_y[:n].cpu()

        memory = list(zip(list(mem_x), list(mem_y)))
        memory_dataset = _make_taskaware_classification_dataset(memory, targets=mem_y.tolist())

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            memory_dataset,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle,
        )

    def after_forward(self, strategy, num_workers: int = 0, shuffle: bool = True, **kwargs):
        """Select samples to fill/replace the memory buffer based on cosine similarity."""
        strategy.model.eval()

        # Kept for API compatibility with older code paths.
        grad_dims = [p.data.numel() for p in strategy.model.parameters()]

        place_left = self.ext_mem_list_x.size(0) - self.ext_mem_list_current_index

        if place_left <= 0:
            # Buffer full: compute batch similarity to decide replacement.
            batch_sim, mem_grads = self.get_batch_sim(
                strategy, grad_dims, batch_x=strategy.mb_x, batch_y=strategy.mb_y
            )

            if batch_sim.item() < 0:
                current_n = self.ext_mem_list_current_index
                mb_size = int(strategy.mb_x.size(0))
                draw_k = min(mb_size, current_n)
                if draw_k == 0:
                    strategy.model.train()
                    return

                buffer_score = self.buffer_score[:current_n].cpu()
                # Normalize to [0,1] for multinomial weights
                denom = (buffer_score.max() - buffer_score.min()) + 0.01
                buffer_sim = (buffer_score - buffer_score.min()) / denom

                # CPU indices for CPU buffer.
                index_cpu = torch.multinomial(buffer_sim, draw_k, replacement=False)

                # Similarity per sample in current batch (on device).
                batch_item_sim = self.get_each_batch_sample_sim(
                    strategy, grad_dims, mem_grads, strategy.mb_x, strategy.mb_y
                )  # [B]

                # Match sizes to draw_k if mb_size > current_n.
                batch_item_sim = batch_item_sim[:draw_k]

                scaled_batch_item_sim = ((batch_item_sim + 1) / 2).unsqueeze(1)  # [K,1]
                buffer_repl_batch_sim = ((self.buffer_score[index_cpu] + 1) / 2).unsqueeze(1)  # [K,1] (CPU)

                # Sample replacement decision per candidate.
                outcome = torch.multinomial(
                    torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim.to(strategy.device)), dim=1),
                    1,
                    replacement=False,
                )  # [K,1]

                # outcome==1 means choose buffer item (keep), outcome==0 means choose new sample (replace)
                replace_mask = outcome.squeeze(1).eq(0)  # [K]
                if replace_mask.any():
                    # Write into CPU buffer using CPU indices; source from GPU minibatch -> CPU.
                    repl_idx_cpu = index_cpu[replace_mask.to(torch.device("cpu"))]
                    src_idx = torch.arange(draw_k, device=strategy.device)[replace_mask]

                    self.ext_mem_list_x[repl_idx_cpu] = strategy.mb_x[src_idx].detach().cpu()
                    self.ext_mem_list_y[repl_idx_cpu] = strategy.mb_y[src_idx].detach().cpu()
                    self.buffer_score[repl_idx_cpu] = batch_item_sim[src_idx].detach().cpu()

        else:
            # Buffer not full: append up to available space.
            offset = min(int(place_left), int(strategy.mb_x.size(0)))
            updated_mb_x = strategy.mb_x[:offset]
            updated_mb_y = strategy.mb_y[:offset]

            if self.ext_mem_list_current_index == 0:
                batch_sample_memory_cos = torch.zeros(offset, device=strategy.device) + 0.1
            else:
                mem_grads = self.get_rand_mem_grads(
                    strategy=strategy, _grad_dims=grad_dims, gss_batch_size=len(strategy.mb_x)
                )
                batch_sample_memory_cos = self.get_each_batch_sample_sim(
                    strategy, grad_dims, mem_grads, updated_mb_x, updated_mb_y
                )  # [offset]

            curr_idx = self.ext_mem_list_current_index
            # Copy GPU -> CPU safely.
            self.ext_mem_list_x[curr_idx : curr_idx + offset].copy_(updated_mb_x.detach().cpu())
            self.ext_mem_list_y[curr_idx : curr_idx + offset].copy_(updated_mb_y.detach().cpu())
            self.buffer_score[curr_idx : curr_idx + offset].copy_(batch_sample_memory_cos.detach().cpu())

            self.ext_mem_list_current_index += offset

        strategy.model.train()
