import torchvision
import wandb
import numpy as np
import torch
from random import randint


def get_adversarial_label(y_true: int, scores: torch.Tensor, previous_labels: list, policy="random") -> int:
    if len(previous_labels) >= len(scores):
        raise AttributeError("No more labels available")
    if policy == "random":
        lab = y_true
        n_classes = len(scores)
        while (y_true == lab) or (lab in previous_labels):
            lab = randint(0, n_classes - 1)
        return lab
    elif policy == "best":
        n = len(previous_labels)
        sorted_scores = scores.argsort(dim=1, descending=True)[n + 1]
        assert sorted_scores[0] == y_true
        return sorted_scores[1].item()
    else:
        AttributeError("policy must be 'random' or 'best'")


class UnNormalize(object):
    """Unormalize a torch tensor

    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for frame in tensor:
            for t, m, s in zip(frame, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
        return tensor


def get_wandb_image(grid, y_target, y_true, diff, unnormalize=False):
    unnorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    grid = grid.cpu().detach()
    if unnormalize:
        grid = unnorm(grid)
    grid = (grid * 255).byte()
    grid = grid.numpy().transpose(1, 2, 0)
    return wandb.Image(grid, caption=f"Adv:{y_target}, True:{y_true}, Diff:{diff:.2f}")


class AdversarialLogger:
    def __init__(self, trainer, policy):
        self.trainer = trainer
        self.policy = policy
        self.times_table = wandb.Table(columns=['idx', "policy", "presolve_strategy", "setup_time", "opt_time"])

    def log_adversarial(self, index, original_image, adv_images, y_advs, y_true, diff_sums):
        diffs = [torch.abs(adv_img - original_image) for adv_img, y_adv in zip(adv_images, y_advs)]

        grid_images = [torchvision.utils.make_grid([original_image, adv_img, diff], padding=5, pad_value=1) for
                       adv_img, diff in zip(adv_images, diffs)]

        for grid, y_adv, diff in zip(grid_images, y_advs, diff_sums):
            img = get_wandb_image(grid, y_adv, y_true, diff)
            self.trainer.logger.experiment.log({
                "idx": index,
                "y_target": y_adv,
                "imgs": img,
                "diff": diff,
                "step": self.trainer.global_step,
                "policy": self.policy
            })

    def add_times(self, index, opt_time, setup_time):
        self.times_table.add_data(index, self.policy, True, setup_time, opt_time)

    def log_times(self):
        wandb.log({
            "times_table": self.times_table,
            "opt_times_line": wandb.plot.line(self.times_table, x="idx", y="opt_time"),
            "setup_times_line": wandb.plot.line(self.times_table, x="idx", y="setup_time")
        })
