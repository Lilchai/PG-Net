import os
from datetime import datetime
import logging
import numpy as np
import nibabel as nib
from skimage.measure import label
from torch import nn
import torch
import torch.nn.functional as F


# ===========================
# Utility functions (kept original)
# ===========================
def setup_logging(fold, log_dir="logs"):
    """设置日志配置"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"train_fold{fold}_{timestamp}.log")

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return log_filename


def save_nifti(data, affine, filename, dtype=None):
    """保存数据为nifti文件，自动处理有问题的仿射矩阵"""
    if dtype is not None:
        data = data.astype(dtype)

    fixed_affine = fix_affine_matrix(affine)

    img = nib.Nifti1Image(data, fixed_affine)
    nib.save(img, filename)


def fix_affine_matrix(affine):
    affine = np.array(affine, dtype=np.float64)

    if affine.shape != (4, 4):
        affine = np.eye(4)

    if np.linalg.det(affine[:3, :3]) == 0:
        print("警告：检测到奇异的仿射矩阵，使用单位矩阵替代")
        return np.eye(4)

    if np.any(np.isnan(affine)) or np.any(np.isinf(affine)):
        print("警告：检测到NaN或inf在仿射矩阵中，使用单位矩阵替代")
        return np.eye(4)

    return affine


def calculate_connected_components_metrics(pred_mask, true_mask):
    """
    基于连通分量计算召回率和误报率
    """
    pred_binary = (pred_mask > 0).astype(np.uint8)
    true_binary = (true_mask > 0).astype(np.uint8)

    pred_components = label(pred_binary, connectivity=3)
    true_components = label(true_binary, connectivity=3)

    pred_labels = np.unique(pred_components)
    pred_labels = pred_labels[pred_labels > 0]

    true_labels = np.unique(true_components)
    true_labels = true_labels[true_labels > 0]

    if len(true_labels) == 0:
        recall = 1.0 if len(pred_labels) == 0 else 0.0
        false_positive_rate = 1.0 if len(pred_labels) > 0 else 0.0
        return recall, false_positive_rate, 0, len(pred_labels), 0

    if len(pred_labels) == 0:
        return 0.0, 0.0, 0, 0, len(true_labels)

    pred_matched = []
    true_matched = set()

    for pred_label in pred_labels:
        pred_component_mask = (pred_components == pred_label)
        matched = False
        for true_label in true_labels:
            if true_label in true_matched:
                continue
            true_component_mask = (true_components == true_label)
            if np.any(np.logical_and(pred_component_mask, true_component_mask)):
                matched = True
                true_matched.add(true_label)
                break
        pred_matched.append(matched)

    tp_count = sum(pred_matched)
    fp_count = len(pred_labels) - tp_count
    fn_count = len(true_labels) - len(true_matched)
    recall = len(true_matched) / len(true_labels)
    false_positive_rate = fp_count / len(pred_labels)

    return recall, false_positive_rate, tp_count, fp_count, fn_count


class UnifiedBoundarySDFLoss3D(nn.Module):
    def __init__(self, clip_dist=5.0):
        """
        Args:
            clip_dist (float): 截断距离，控制边界缓冲区的范围
        """
        super().__init__()
        self.clip_dist = clip_dist

    def forward(self, pred, gt, sdf):
        # 确保类型正确
        pred = pred.as_tensor().float()
        gt = gt.as_tensor().float()
        sdf = sdf.as_tensor().float()

        sdf_clipped = torch.clamp(sdf, min=-self.clip_dist, max=0)
        
        boundary_loss_field = (pred - gt) * sdf_clipped

        boundary_mask = (sdf >= -self.clip_dist) & (sdf <= 0)

        if boundary_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        loss = boundary_loss_field[boundary_mask].mean()

        return loss