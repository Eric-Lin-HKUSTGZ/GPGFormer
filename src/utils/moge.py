import torch
from ..moge.model.v2 import MoGeModel
from ..moge.utils.geometry_torch import normalized_view_plane_uv


def _infer_neck_out_channels(model: MoGeModel) -> int:
    neck = getattr(model, "neck", None)
    if neck is None:
        return 256
    output_blocks = getattr(neck, "output_blocks", None)
    if output_blocks:
        last_block = output_blocks[-1]
        if hasattr(last_block, "out_channels"):
            return int(last_block.out_channels)
    res_blocks = getattr(neck, "res_blocks", None)
    if res_blocks:
        last_seq = res_blocks[-1]
        for layer in reversed(list(last_seq.modules())):
            if hasattr(layer, "out_channels"):
                return int(layer.out_channels)
    return 256


class MoGeFeatureExtractor(torch.nn.Module):
    def __init__(self, checkpoint: str, num_tokens: int = 2500, use_fp16: bool = True):
        super().__init__()
        self.model = MoGeModel.from_pretrained(checkpoint)
        self.num_tokens = num_tokens
        self.use_fp16 = use_fp16
        self.out_channels = _infer_neck_out_channels(self.model)

    def to(self, device: torch.device):
        self.model = self.model.to(device)
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(device=self.model.device, dtype=self.model.dtype)

        batch_size, _, img_h, img_w = image.shape
        aspect_ratio = img_w / float(img_h)
        base_h = int(round((self.num_tokens / aspect_ratio) ** 0.5))
        base_w = int(round((self.num_tokens * aspect_ratio) ** 0.5))

        if self.use_fp16 and image.device.type == "cuda" and image.dtype != torch.float16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features, _ = self.model.encoder(image, base_h, base_w, return_class_token=True)
                features = [features, None, None, None, None]
                for level in range(5):
                    uv = normalized_view_plane_uv(
                        width=base_w * 2 ** level,
                        height=base_h * 2 ** level,
                        aspect_ratio=aspect_ratio,
                        dtype=image.dtype,
                        device=image.device
                    )
                    uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
                    if features[level] is None:
                        features[level] = uv
                    else:
                        features[level] = torch.concat([features[level], uv], dim=1)
                features = self.model.neck(features)
                return features[-1]

        features, _ = self.model.encoder(image, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2 ** level,
                height=base_h * 2 ** level,
                aspect_ratio=aspect_ratio,
                dtype=image.dtype,
                device=image.device
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        features = self.model.neck(features)
        return features[-1]
