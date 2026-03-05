# -*- coding: utf-8 -*-
"""
Exponential Moving Average (EMA) for model parameters.
"""
import copy
from typing import Optional

import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average (EMA) for model parameters.

    Maintains a shadow copy of model parameters that are updated using
    exponential moving average. This can improve model generalization
    and stability during inference.

    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.9999)
        device: Device to store EMA parameters (default: None, use model's device)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None
    ):
        self.decay = decay
        self.device = device
        self._updates = 0

        # Create a shadow copy of the model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Move to specified device if provided
        if device is not None:
            self.ema_model.to(device)

        # Disable gradient computation for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

        # Cache which state_dict keys are parameters vs buffers.
        # Apply EMA to parameters and copy buffers directly (e.g., BatchNorm running stats).
        self._param_keys = {k for k, _ in self.ema_model.named_parameters()}
        self._buffer_keys = {k for k, _ in self.ema_model.named_buffers()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA parameters using current model parameters.

        EMA update rule:
            ema_param = decay * ema_param + (1 - decay) * model_param

        Args:
            model: Current model with updated parameters
        """
        self._updates += 1

        model_state = model.state_dict()
        ema_state = self.ema_model.state_dict()

        decay = float(self.decay)
        one_minus = 1.0 - decay

        # Update EMA in-place to avoid per-step load_state_dict overhead.
        for key, ema_v in ema_state.items():
            model_v = model_state.get(key, None)
            if model_v is None:
                continue

            model_v = model_v.detach()
            if model_v.device != ema_v.device:
                model_v = model_v.to(device=ema_v.device)

            if key in self._param_keys and torch.is_floating_point(ema_v):
                ema_v.mul_(decay).add_(model_v, alpha=one_minus)
            else:
                ema_v.copy_(model_v)

    def state_dict(self):
        """Return EMA model's state dict."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict into EMA model."""
        self.ema_model.load_state_dict(state_dict)

    def module(self):
        """Return the EMA model."""
        return self.ema_model
