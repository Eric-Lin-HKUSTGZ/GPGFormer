"""
GPGFormer Inference Script for FreiHAND Evaluation
Similar to simpleHand/infer_to_json.py, computes comprehensive metrics including:
- 3D KP (keypoint) metrics: AUC, mean error
- 3D KP ALIGNED metrics: AUC, mean error (Procrustes aligned)
- 3D MESH metrics: AUC, mean error
- 3D MESH ALIGNED metrics: AUC, mean error
- F-scores at 5mm and 15mm thresholds
"""

import json
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from scipy.linalg import orthogonal_procrustes
import open3d as o3d

from gpgformer.models import GPGFormer, GPGFormerConfig


class EvalUtil:
    """ Util class for evaluation networks. """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred, skip_check=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if not skip_check:
            keypoint_gt = np.squeeze(keypoint_gt)
            keypoint_pred = np.squeeze(keypoint_pred)
            keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

            assert len(keypoint_gt.shape) == 2
            assert len(keypoint_pred.shape) == 2
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds


def verts2pcd(verts, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    if color is not None:
        if color == 'r':
            pcd.paint_uniform_color([1, 0.0, 0])
        if color == 'g':
            pcd.paint_uniform_color([0, 1.0, 0])
        if color == 'b':
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall


def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t


def align_by_trafo(verts, trafo):
    """Apply transformation to vertices."""
    R, s, s1, t = trafo
    verts_t = verts - t
    verts_t = verts_t / s1
    verts_t = verts_t / s
    verts_t = np.dot(verts_t, R)
    verts_t = verts_t * s
    verts_t = verts_t * s1
    return verts_t


def build_dataset(cfg: dict):
    """Build FreiHAND test dataset."""
    name = cfg["dataset"]["name"].lower()

    if name in ("freihand",):
        from data.freihand_dataset import FreiHANDDataset

        return FreiHANDDataset(
            root_dir=cfg["paths"]["freihand_root"],
            eval_root=cfg["paths"].get("freihand_eval_root", None),
            img_size=int(cfg["dataset"].get("img_size", 256)),
            train=False,  # Use test/val split
            align_wilor_aug=bool(cfg["dataset"].get("align_wilor_aug", True)),
            wilor_aug_config=cfg["dataset"].get("wilor_aug_config", {}),
            bbox_source=cfg["dataset"].get("bbox_source_eval", "detector"),
            detector_weights_path=cfg["paths"].get("detector_ckpt", None),
            use_trainval_split=bool(cfg["dataset"].get("use_trainval_split", True)),
            trainval_ratio=float(cfg["dataset"].get("trainval_ratio", 0.9)),
            trainval_seed=int(cfg["dataset"].get("trainval_seed", 42)),
        )

    raise ValueError(f"Unknown dataset.name: {cfg['dataset']['name']}")


def build_model_from_cfg(cfg: dict) -> GPGFormer:
    model_cfg = cfg.get("model", {})
    refiner_cfg = model_cfg.get("feature_refiner", {})
    moge2_num_tokens = int(model_cfg.get("moge2_num_tokens", 400))
    if moge2_num_tokens <= 0:
        raise ValueError(f"model.moge2_num_tokens must be a positive int, got {moge2_num_tokens}")

    return GPGFormer(
        GPGFormerConfig(
            wilor_ckpt_path=cfg["paths"]["wilor_ckpt"],
            moge2_weights_path=cfg["paths"].get("moge2_ckpt", None),
            use_geo_prior=bool(model_cfg.get("use_geo_prior", True)),
            mano_model_path=cfg["paths"]["mano_dir"],
            mano_mean_params=cfg["paths"]["mano_mean_params"],
            mano_decoder=str(model_cfg.get("mano_decoder", "wilor")),
            freihand_mano_root=model_cfg.get("freihand_mano_root", None),
            focal_length=float(model_cfg.get("focal_length", 5000.0)),
            mano_head_ief_iters=int(model_cfg.get("mano_head", {}).get("ief_iters", 3)),
            mano_head_transformer_input=str(model_cfg.get("mano_head", {}).get("transformer_input", "mean_shape")),
            mano_head_dim=int(model_cfg.get("mano_head", {}).get("dim", 1024)),
            mano_head_depth=int(model_cfg.get("mano_head", {}).get("depth", 6)),
            mano_head_heads=int(model_cfg.get("mano_head", {}).get("heads", 8)),
            mano_head_dim_head=int(model_cfg.get("mano_head", {}).get("dim_head", 64)),
            mano_head_mlp_dim=int(model_cfg.get("mano_head", {}).get("mlp_dim", 2048)),
            mano_head_dropout=float(model_cfg.get("mano_head", {}).get("dropout", 0.0)),
            moge2_num_tokens=moge2_num_tokens,
            moge2_output=str(model_cfg.get("moge2_output", "neck")),
            token_fusion_mode=str(model_cfg.get("token_fusion_mode", "concat")),
            sum_fusion_strategy=str(model_cfg.get("sum_fusion_strategy", "basic")),
            fusion_proj_zero_init=bool(model_cfg.get("fusion_proj_zero_init", True)),
            cross_attn_num_heads=int(model_cfg.get("cross_attn_num_heads", 8)),
            cross_attn_dropout=float(model_cfg.get("cross_attn_dropout", 0.0)),
            cross_attn_gate_init=float(model_cfg.get("cross_attn_gate_init", 0.0)),
            geo_tokenizer_use_pooling=bool(model_cfg.get("geo_tokenizer_use_pooling", True)),
            feature_refiner_method=str(refiner_cfg.get("method", "none")),
            feature_refiner_feat_dim=int(refiner_cfg.get("feat_dim", 1280)),
            feature_refiner_sjta_bottleneck_dim=int(refiner_cfg.get("sjta_bottleneck_dim", 256)),
            feature_refiner_sjta_num_heads=int(refiner_cfg.get("sjta_num_heads", 4)),
            feature_refiner_sjta_use_2d_prior=bool(refiner_cfg.get("sjta_use_2d_prior", True)),
            feature_refiner_sjta_num_steps=int(refiner_cfg.get("sjta_num_steps", 2)),
            feature_refiner_coear_dilation1=int(refiner_cfg.get("coear_dilation1", 1)),
            feature_refiner_coear_dilation2=int(refiner_cfg.get("coear_dilation2", 2)),
            feature_refiner_coear_gate_reduction=int(refiner_cfg.get("coear_gate_reduction", 8)),
            feature_refiner_coear_init_alpha=float(refiner_cfg.get("coear_init_alpha", 0.1)),
            feature_refiner_wilor_msf_bottleneck_ratio=int(refiner_cfg.get("wilor_msf_bottleneck_ratio", 4)),
            feature_refiner_wilor_msf_dilation1=int(refiner_cfg.get("wilor_msf_dilation1", 1)),
            feature_refiner_wilor_msf_dilation2=int(refiner_cfg.get("wilor_msf_dilation2", 2)),
            feature_refiner_wilor_msf_dilation3=int(refiner_cfg.get("wilor_msf_dilation3", 3)),
            feature_refiner_wilor_msf_gate_reduction=int(refiner_cfg.get("wilor_msf_gate_reduction", 8)),
            feature_refiner_wilor_msf_init_alpha=float(refiner_cfg.get("wilor_msf_init_alpha", 0.1)),
            feature_refiner_kcr_num_keypoints=int(refiner_cfg.get("kcr_num_keypoints", 21)),
            feature_refiner_kcr_hidden_dim=int(refiner_cfg.get("kcr_hidden_dim", 128)),
        )
    )


def warmup_lazy_modules(model: GPGFormer, cfg: dict, device: torch.device) -> None:
    model_cfg = cfg.get("model", {})
    h = int(model_cfg.get("image_size", 256))
    w = int(model_cfg.get("image_width", int(h * 0.75)))
    # Avoid `torch.inference_mode()` here: lazily-created parameters become "inference tensors"
    # and `load_state_dict()` (which does in-place copies) will error outside InferenceMode.
    with torch.no_grad():
        img_dummy = torch.zeros((1, 3, h, w), device=device, dtype=torch.float32)
        cam_dummy = torch.tensor([[600.0, 600.0, w / 2.0, h / 2.0]], device=device, dtype=torch.float32)
        _ = model(img_dummy, cam_param=cam_dummy)


def infer_single_json(cfg, model, device):
    """Run inference on the test dataset and collect predictions."""
    dataset = build_dataset(cfg)
    sampler = SequentialSampler(dataset)
    batch_size = int(cfg["train"].get("val_batch_size", 16))
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    pred_joints_list = []
    pred_vertices_list = []
    gt_joints_list = []
    gt_vertices_list = []

    print(f"Running inference on {len(dataset)} samples...")

    for batch_data in tqdm(dataloader, desc="Inference"):
        img = batch_data["rgb"].to(device)
        cam_param = batch_data.get("cam_param", None)
        cam_param = cam_param.to(device) if cam_param is not None else None

        with torch.no_grad():
            out = model(img, cam_param=cam_param)

        # Get predictions in camera coordinates (mm)
        pred_cam_t = out["pred_cam_t"]  # (B, 3)
        pred_keypoints_3d = out["pred_keypoints_3d"]  # (B, 21, 3) in meters
        pred_vertices = out["pred_vertices"]  # (B, 778, 3) in meters

        # Convert to mm and add camera translation
        pred_joints_mm = (pred_keypoints_3d + pred_cam_t.unsqueeze(1)) * 1000.0
        pred_verts_mm = (pred_vertices + pred_cam_t.unsqueeze(1)) * 1000.0

        # Get ground truth (already in mm)
        gt_joints_mm = batch_data["keypoints_3d"]  # (B, 21, 3)
        try:
            gt_verts_mm = batch_data.get("vertices_gt", None)  # (B, 778, 3) or None
        except KeyError:
            gt_verts_mm = None
        # Convert to meters for consistency with evaluation
        pred_joints_list.append(pred_joints_mm.cpu().numpy() / 1000.0)
        pred_vertices_list.append(pred_verts_mm.cpu().numpy() / 1000.0)
        gt_joints_list.append(gt_joints_mm.cpu().numpy() / 1000.0)

        if gt_verts_mm is not None:
            gt_vertices_list.append(gt_verts_mm.cpu().numpy() / 1000.0)

    # Concatenate all batches
    pred_joints_list = np.concatenate(pred_joints_list, axis=0)
    pred_vertices_list = np.concatenate(pred_vertices_list, axis=0)
    gt_joints_list = np.concatenate(gt_joints_list, axis=0)

    if len(gt_vertices_list) > 0:
        gt_vertices_list = np.concatenate(gt_vertices_list, axis=0)
    else:
        gt_vertices_list = None

    return pred_joints_list, pred_vertices_list, gt_joints_list, gt_vertices_list


def main():
    parser = argparse.ArgumentParser(description="GPGFormer Inference and Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (optional)")
    args = parser.parse_args()

    # Load config
    cfg = yaml.safe_load(Path(args.config).read_text())
    print(f"Loaded config from {args.config}")
    model_cfg = cfg.get("model", {})
    moge2_num_tokens = int(model_cfg.get("moge2_num_tokens", 400))
    if moge2_num_tokens <= 0:
        raise ValueError(f"model.moge2_num_tokens must be a positive int, got {moge2_num_tokens}")
    moge2_output = str(model_cfg.get("moge2_output", "neck"))
    use_geo_prior = bool(model_cfg.get("use_geo_prior", True))
    print(f"[info] use_geo_prior={use_geo_prior} moge2_output={moge2_output} moge2_num_tokens={moge2_num_tokens}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    print("Building model...")
    model = build_model_from_cfg(cfg).to(device)
    # Ensure lazily created modules (e.g., geo_tokenizer) exist before loading checkpoint.
    warmup_lazy_modules(model, cfg, device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    print("Model loaded successfully")

    # Run inference
    xyz_pred_list, verts_pred_list, xyz_gt_list, verts_gt_list = infer_single_json(cfg, model, device)

    print(f"\nCollected {len(xyz_pred_list)} samples")
    print(f"Pred joints shape: {xyz_pred_list.shape}")
    print(f"Pred vertices shape: {verts_pred_list.shape}")
    print(f"GT joints shape: {xyz_gt_list.shape}")
    if verts_gt_list is not None:
        print(f"GT vertices shape: {verts_gt_list.shape}")

    # Evaluate metrics
    print("\n" + "="*60)
    print("Computing evaluation metrics...")
    print("="*60)

    eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
    eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)
    f_score, f_score_aligned = list(), list()
    f_threshs = [0.005, 0.015]
    shape_is_mano = None

    for idx in range(len(xyz_gt_list)):
        xyz, verts = xyz_gt_list[idx], verts_gt_list[idx] if verts_gt_list is not None else None
        xyz_pred, verts_pred = xyz_pred_list[idx], verts_pred_list[idx]

        # Not aligned errors
        eval_xyz.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred
        )

        if shape_is_mano is None and verts is not None:
            if verts_pred.shape[0] == verts.shape[0]:
                shape_is_mano = True
            else:
                shape_is_mano = False

        if shape_is_mano and verts is not None:
            eval_mesh_err.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred
            )

        # align predictions
        xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
        if shape_is_mano and verts is not None:
            verts_pred_aligned = align_w_scale(verts, verts_pred)
        elif verts is not None:
            # use trafo estimated from keypoints
            trafo = align_w_scale(xyz, xyz_pred, return_trafo=True)
            verts_pred_aligned = align_by_trafo(verts_pred, trafo)
        else:
            verts_pred_aligned = None

        # Aligned errors
        eval_xyz_aligned.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred_aligned
        )

        if shape_is_mano and verts is not None:
            eval_mesh_err_aligned.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred_aligned
            )

        # F-scores
        if verts is not None and verts_pred_aligned is not None:
            l, la = list(), list()
            for t in f_threshs:
                # for each threshold calculate the f score and the f score of the aligned vertices
                f, _, _ = calculate_fscore(verts, verts_pred, t)
                l.append(f)
                f, _, _ = calculate_fscore(verts, verts_pred_aligned, t)
                la.append(f)
            f_score.append(l)
            f_score_aligned.append(la)

    # Calculate and print results
    print("\n" + "="*60)
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

    xyz_al_mean3d, _, xyz_al_auc3d, pck_xyz_al, thresh_xyz_al = eval_xyz_aligned.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP ALIGNED results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (xyz_al_auc3d, xyz_al_mean3d * 100.0))

    if shape_is_mano and verts_gt_list is not None:
        mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (mesh_auc3d, mesh_mean3d * 100.0))

        mesh_al_mean3d, _, mesh_al_auc3d, pck_mesh_al, thresh_mesh_al = eval_mesh_err_aligned.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH ALIGNED results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (mesh_al_auc3d, mesh_al_mean3d * 100.0))
    else:
        mesh_mean3d, mesh_auc3d, mesh_al_mean3d, mesh_al_auc3d = -1.0, -1.0, -1.0, -1.0

    if len(f_score) > 0:
        print('F-scores')
        f_score, f_score_aligned = np.array(f_score).T, np.array(f_score_aligned).T
        for f, fa, t in zip(f_score, f_score_aligned, f_threshs):
            print('F@%.1fmm = %.4f' % (t*1000, f.mean()), '\tF_aligned@%.1fmm = %.4f' % (t*1000, fa.mean()))

    print("="*60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
