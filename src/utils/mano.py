import pickle
from typing import Optional

import torch
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import to_tensor
from smplx.vertex_ids import vertex_ids


WILOR_JOINT_MAP = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


class MANO(smplx.MANOLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if joint_regressor_extra is not None:
            self.register_buffer(
                "joint_regressor_extra",
                torch.tensor(pickle.load(open(joint_regressor_extra, "rb"), encoding="latin1"), dtype=torch.float32),
            )
        self.register_buffer("extra_joints_idxs", to_tensor(list(vertex_ids["mano"].values()), dtype=torch.long))
        self.register_buffer("joint_map", torch.tensor(WILOR_JOINT_MAP, dtype=torch.long))

    def forward(self, *args, **kwargs):
        mano_output = super().forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, "joint_regressor_extra"):
            extra_joints = vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        mano_output.joints = joints
        return mano_output
