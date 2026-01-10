from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from third_party.wilor_min.wilor.models.mano_wrapper import MANO


@dataclass(frozen=True)
class MANOConfig:
    model_path: str
    mean_params: str
    gender: str = "neutral"
    num_hand_joints: int = 15


class MANOLayer(nn.Module):
    def __init__(self, cfg: MANOConfig):
        super().__init__()
        model_path = Path(cfg.model_path)
        mean_params = Path(cfg.mean_params)
        if not model_path.exists():
            raise FileNotFoundError(f"MANO model_path not found: {model_path}")
        if not mean_params.exists():
            raise FileNotFoundError(f"MANO mean params not found: {mean_params}")

        # WiLoR's MANO wrapper expects smplx MANO assets in model_path
        self.mano = MANO(
            model_path=str(model_path),
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=False,
            num_pca_comps=0,
            joint_regressor_extra=None,
        )

    def forward(self, mano_params: dict, pose2rot: bool = False):
        """
        Args:
            mano_params: dict with keys: global_orient (B,1,3,3), hand_pose (B,15,3,3), betas (B,10)
        Returns:
            mano_output with .vertices (B,V,3) and .joints (B,21,3) in MANO canonical coordinates.
        """
        return self.mano(
            global_orient=mano_params["global_orient"],
            hand_pose=mano_params["hand_pose"],
            betas=mano_params["betas"],
            pose2rot=pose2rot,
        )


