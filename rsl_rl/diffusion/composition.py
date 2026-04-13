from __future__ import annotations

import torch


_G1_LOWER_BODY_JOINTS = {
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
}
_G1_SHARED_BODY_JOINTS = {"waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"}
_G1_UPPER_BODY_JOINTS = {
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}
_G1_LOWER_BODY_EE = {"left_ankle_roll_link", "right_ankle_roll_link"}
_G1_UPPER_BODY_EE = {"left_wrist_roll_link", "right_wrist_roll_link"}
_G1_SHARED_KEY_BODIES = {"pelvis", "torso_link"}
_G1_LOWER_KEY_BODIES = {
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
}
_G1_UPPER_KEY_BODIES = {
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
}


def _ensure_feature_block(feature_block_offsets: dict[str, tuple[int, int]], key: str) -> tuple[int, int]:
    if key not in feature_block_offsets:
        raise KeyError(f"Missing feature block offset for '{key}'")
    start, stop = feature_block_offsets[key]
    return int(start), int(stop)


def _find_joint_feature_block(feature_block_offsets: dict[str, tuple[int, int]]) -> tuple[str, tuple[int, int]]:
    for key in ("joint_rot6d_rel", "joint_pos_rel"):
        if key in feature_block_offsets:
            return key, tuple(map(int, feature_block_offsets[key]))
    raise KeyError("Missing feature block offset for either 'joint_rot6d_rel' or 'joint_pos_rel'")


def _validate_masks(feature_masks: dict[str, torch.Tensor]) -> None:
    if not feature_masks:
        raise ValueError("feature_masks must be non-empty")
    stacked = torch.stack([mask.float() for mask in feature_masks.values()], dim=0)
    coverage = stacked.sum(dim=0)
    if not torch.allclose(coverage, torch.ones_like(coverage)):
        raise ValueError("feature masks must be mutually exclusive and fully cover all features")


def build_g1_body_part_feature_masks(
    mask_name: str,
    joint_name_order: list[str],
    ee_name_order: list[str],
    key_body_name_order: list[str],
    feature_block_offsets: dict[str, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """根据 G1 特征布局构建 shared/lower/upper 三组掩码。"""
    if mask_name != "g1_upper_lower":
        raise ValueError(f"Unsupported mask template: {mask_name}")

    feature_dim = max(stop for _, stop in feature_block_offsets.values())
    masks = {
        "shared_body": torch.zeros(feature_dim, dtype=torch.float32),
        "lower_body": torch.zeros(feature_dim, dtype=torch.float32),
        "upper_body": torch.zeros(feature_dim, dtype=torch.float32),
    }

    base_lin_start, base_lin_stop = _ensure_feature_block(feature_block_offsets, "base_lin_vel_b")
    base_ang_start, base_ang_stop = _ensure_feature_block(feature_block_offsets, "base_ang_vel_b")
    masks["shared_body"][base_lin_start:base_lin_stop] = 1.0
    masks["shared_body"][base_ang_start:base_ang_stop] = 1.0

    _, (joint_start, joint_stop) = _find_joint_feature_block(feature_block_offsets)
    joint_block_size = joint_stop - joint_start
    if joint_block_size % len(joint_name_order) != 0:
        raise ValueError("Joint feature block size does not divide evenly by joint_name_order")
    joint_width = joint_block_size // len(joint_name_order)
    for offset, joint_name in enumerate(joint_name_order):
        start = joint_start + offset * joint_width
        stop = start + joint_width
        if joint_name in _G1_LOWER_BODY_JOINTS:
            masks["lower_body"][start:stop] = 1.0
        elif joint_name in _G1_SHARED_BODY_JOINTS:
            masks["shared_body"][start:stop] = 1.0
        elif joint_name in _G1_UPPER_BODY_JOINTS:
            masks["upper_body"][start:stop] = 1.0
        else:
            raise ValueError(f"Unknown G1 joint in mask template: {joint_name}")

    ee_start, ee_stop = _ensure_feature_block(feature_block_offsets, "ee_pos_b")
    ee_width = (ee_stop - ee_start) // len(ee_name_order)
    for offset, ee_name in enumerate(ee_name_order):
        start = ee_start + offset * ee_width
        stop = start + ee_width
        if ee_name in _G1_LOWER_BODY_EE:
            masks["lower_body"][start:stop] = 1.0
        elif ee_name in _G1_UPPER_BODY_EE:
            masks["upper_body"][start:stop] = 1.0
        else:
            raise ValueError(f"Unknown G1 end-effector in mask template: {ee_name}")

    if "key_body_rot6d" in feature_block_offsets:
        key_start, key_stop = _ensure_feature_block(feature_block_offsets, "key_body_rot6d")
        key_width = (key_stop - key_start) // len(key_body_name_order)
        for offset, body_name in enumerate(key_body_name_order):
            start = key_start + offset * key_width
            stop = start + key_width
            if body_name in _G1_SHARED_KEY_BODIES:
                masks["shared_body"][start:stop] = 1.0
            elif body_name in _G1_LOWER_KEY_BODIES:
                masks["lower_body"][start:stop] = 1.0
            elif body_name in _G1_UPPER_KEY_BODIES:
                masks["upper_body"][start:stop] = 1.0
            else:
                raise ValueError(f"Unknown G1 key body in mask template: {body_name}")

    _validate_masks(masks)
    return masks


def compose_style_predictions_with_body_masks(
    part_to_eps: dict[str, torch.Tensor],
    feature_masks: dict[str, torch.Tensor],
) -> torch.Tensor:
    """按身体部位掩码组合多个 style-conditioned epsilon 预测。"""
    if not part_to_eps:
        raise ValueError("part_to_eps must be non-empty")
    if set(part_to_eps) != set(feature_masks):
        raise ValueError("part_to_eps and feature_masks must share the same keys")

    reference = next(iter(part_to_eps.values()))
    composed = torch.zeros_like(reference)
    for part_name, eps_part in part_to_eps.items():
        if eps_part.shape != reference.shape:
            raise ValueError("All epsilon predictions must share the same shape")
        mask = feature_masks[part_name].to(device=eps_part.device, dtype=eps_part.dtype)
        if mask.ndim == 1:
            mask = mask.view(1, 1, -1)
        if mask.shape[-1] != eps_part.shape[-1]:
            raise ValueError(
                f"Mask feature dim {mask.shape[-1]} does not match epsilon feature dim {eps_part.shape[-1]}"
            )
        composed = composed + eps_part * mask
    return composed
