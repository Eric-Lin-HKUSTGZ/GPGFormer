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

        # Create a shadow copy of the model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Move to specified device if provided
        if device is not None:
            self.ema_model.to(device)

        # Disable gradient computation for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA parameters using current model parameters.

        EMA update rule:
            ema_param = decay * ema_param + (1 - decay) * model_param

        Args:
            model: Current model with updated parameters
        """
        # Get state dicts
        model_state = model.state_dict()
        ema_state = self.ema_model.state_dict()

        # Update EMA parameters
        for key in model_state.keys():
            if key in ema_state:
                ema_param = ema_state[key]
                model_param = model_state[key]

                # Only update floating point parameters
                if ema_param.dtype in [torch.float16, torch.float32, torch.float64]:
                    ema_state[key] = (
                        self.decay * ema_param + (1.0 - self.decay) * model_param
                    )
                else:
                    # For non-floating point parameters (e.g., buffers), just copy
                    ema_state[key] = model_param.clone()

        # Load updated state dict
        self.ema_model.load_state_dict(ema_state)

    def state_dict(self):
        """Return EMA model's state dict."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict into EMA model."""
        self.ema_model.load_state_dict(state_dict)

    def module(self):
        """Return the EMA model."""
        return self.ema_model
