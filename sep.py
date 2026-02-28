# SeparateReplayLossPlugin.py

import torch
from avalanche.core import SupervisedPlugin

class SeparateReplayLossPlugin(SupervisedPlugin):
    """
    Compute loss as: loss_current + alpha * loss_memory,
    where alpha = 1 / (# classes encountered so far).

    Important:
    - Add this plugin BEFORE ReplayPlugin so we can capture the
      pre-replay minibatch size in before_training_iteration.
    - Works when ReplayPlugin concatenates memory samples after the current
      minibatch (default behavior).
    """

    def __init__(self):
        super().__init__()

    # replace the loss just before backward
    @torch.no_grad()
    def before_backward(self, strategy, **kwargs):
        if len(strategy.mb_y) <= strategy.train_mb_size:
            return
        
        torch.set_grad_enabled(True)

        y_pred = strategy.mb_output
        y_true = strategy.mb_y

        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]

        y_pred_cur = y_pred[strategy.train_mb_size:]
        y_true_cur = y_true[strategy.train_mb_size:]

        y_pred_mem = y_pred[:strategy.train_mb_size]
        y_true_mem = y_true[:strategy.train_mb_size]

        seen = getattr(strategy.experience, "classes_seen_so_far", None)
        if not seen:
            alpha = 0.0
        else:
            alpha = 1.0 / float(len(seen))

        loss_cur = strategy._criterion(y_pred_cur, y_true_cur)
        loss_mem = strategy._criterion(y_pred_mem, y_true_mem)

        new_loss = loss_cur + alpha * loss_mem
        strategy.loss = new_loss


