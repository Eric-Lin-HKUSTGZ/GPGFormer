from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class AttrDict(dict):
    """
    Minimal attribute-access dict (cfg.MODEL.IMAGE_SIZE style) compatible with the vendored WiLoR ViT.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return super().get(key, default)


def to_attr_dict(x: Any) -> Any:
    if isinstance(x, Mapping):
        out = AttrDict()
        for k, v in x.items():
            out[k] = to_attr_dict(v)
        return out
    return x



