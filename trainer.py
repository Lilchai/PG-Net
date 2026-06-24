import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from monai.transforms import AsDiscrete
from monai.data import MetaTensor, decollate_batch, list_data_collate
from monai.losses import DiceLoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_iou
from monai.utils import set_determinism
from monai.utils.enums import MetricReduction


from utilities.utilities import *
from utilities.get_datasets import *
from network.U2Net_with_atn_dif import FMV_Net



def train():
    args = get_args()

    set_determinism(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    log_file = setup_logging(args.fold, args.log_dir)
    logging.info(f"训练日志文件: {log_file}")
    logging.info(f"训练参数: {vars(args)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # ----------- 1. 数据加载函数 -----------
    train_files, val_files, train_transforms, val_transforms = get_datasets_vessel_at(
        args.data_dir, args.fold, args.seed, args.crop_size
    )

    from monai.data import Dataset
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    logging.info(f"训练集大小: {len(train_ds)}, 验证集大小: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=list_data_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)

    # ----------- 2. 模型初始化 -----------
    model = FMV_Net(
        in_channels=1,
        out_channels=args.num_classes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"模型总参数: {total_params:,}")

    # Losses & Optimizer
    dice_fn = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    focal_fn = FocalLoss(to_onehot_y=True, weight=torch.tensor([1.0, 10.0]).to(device))
    boundary_fn = UnifiedBoundarySDFLoss3D().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    post_label = AsDiscrete(to_onehot=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_classes)
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN)

    best_metric = -1
    best_epoch = -1

    if args.save_val_images:
        val_output_dir = os.path.join(args.val_output_dir, f"fold{args.fold}")
        os.makedirs(val_output_dir, exist_ok=True)

    start_time = datetime.now()

    for epoch in range(args.epochs):
        model.train()
        step = 0
        epoch_total_loss = 0
        epoch_dice_loss = 0
        epoch_focal_loss = 0
        epoch_boundary_loss = 0

        for batch_data in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", ncols=100):
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            vessel_map = batch_data["attention"].to(device)
            sdfs = batch_data["sdf"].to(device)

            optimizer.zero_grad()

            mask_pred = model(inputs, vessel_map=vessel_map)

            d_loss = dice_fn(mask_pred, labels)
            f_loss = focal_fn(mask_pred, labels)

            probs = F.softmax(mask_pred, dim=1)

            if probs.shape[1] > 1 and sdfs.shape[1] == 1:
                b_loss = boundary_fn(probs[:, 1:2, ...], labels, sdfs)
            else:
                b_loss = boundary_fn(probs, labels, sdfs)

            loss = d_loss + f_loss + b_loss

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_dice_loss += d_loss.item()
            epoch_focal_loss += f_loss.item()
            epoch_boundary_loss += b_loss.item()

        # 计算平均损失
        epoch_total_loss /= step
        epoch_dice_loss /= step
        epoch_focal_loss /= step
        epoch_boundary_loss /= step

        logging.info(f"Epoch {epoch + 1} -> Avg Total Loss: {epoch_total_loss:.4f} | "
                     f"Dice Loss: {epoch_dice_loss:.4f} | "
                     f"Focal Loss: {epoch_focal_loss:.4f} | "
                     f"Boundary Loss: {epoch_boundary_loss:.4f} | "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ----------- 4. 验证部分 -----------
        do_val = ((epoch + 1) % 20 == 0) or (epoch == args.epochs - 1)
        if do_val:
            model.eval()
            dice_metric.reset()

            # 1. 初始化
            total_inference_time = 0.0
            val_start_time = datetime.now()
            total_recall, total_fpr = 0.0, 0.0
            total_tp, total_fp, total_fn = 0, 0, 0
            val_samples = 0
            total_iou = 0.0

            with torch.no_grad():
                for i, val_data in enumerate(tqdm(val_loader, desc=f"Validate Epoch {epoch + 1}", ncols=100)):
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_vessel = val_data["attention"].to(device)  # 获取验证集注意力图

                    combined_input = torch.cat([val_inputs, val_vessel], dim=1)  # [1, 2, D, H, W]

                    def combined_predictor(x):
                        # x 是滑窗切下来的局部块，形状为 [1, 2, 96, 96, 96]
                        img_slice = x[:, 0:1, ...]  # 拆出第 1 个通道 (原始图像块)
                        vessel_slice = x[:, 1:2, ...]  # 拆出第 2 个通道 (血管引导块)

                        return model(img_slice, vessel_map=vessel_slice)

                    # --- 速度测量 ---
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    start_time = time.time()

                    val_outputs = sliding_window_inference(
                        inputs=combined_input,
                        roi_size=args.crop_size,
                        sw_batch_size=1,
                        predictor=combined_predictor,
                        overlap=0.5,
                        mode="constant",
                    )

                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.time()
                    total_inference_time += (end_time - start_time)

                    del combined_input

                    val_outputs = val_outputs.to(device)

                    # --- 后处理与小物体去除 ---
                    if isinstance(val_outputs, MetaTensor):
                        val_outputs_tensor = val_outputs.as_tensor()
                    else:
                        val_outputs_tensor = val_outputs

                    if isinstance(val_labels, MetaTensor):
                        val_labels_tensor = val_labels.as_tensor()
                    else:
                        val_labels_tensor = val_labels

                    val_outputs_list = [post_pred(i) for i in decollate_batch(val_outputs_tensor)]
                    val_labels_list = [post_label(i) for i in decollate_batch(val_labels_tensor)]


                    try:
                        if hasattr(val_outputs[0], "pixdim"):
                            spacing = val_outputs[0].pixdim[:3] if len(val_outputs[0].pixdim) == 3 else val_outputs[
                                0].pixdim[1:4]
                        elif "image_meta_dict" in val_data:
                            spacing = val_data["image_meta_dict"]["pixdim"][0][1:4]
                        else:
                            spacing = [0.6, 0.5, 0.5]

                        # 修复：确保转换为 numpy 数组以避免 np.prod 报错
                        if torch.is_tensor(spacing):
                            spacing = spacing.cpu().numpy()
                        else:
                            spacing = np.array(spacing)

                        voxel_vol = np.prod(spacing)  # 单个体素的体积 (mm^3)
                    except:
                        spacing = np.array([0.6, 0.5, 0.5])
                        voxel_vol = np.prod(spacing)

                    min_voxel_count = int(np.ceil(1.0 / voxel_vol))  # 1mm^3 对应体素数

                    # 执行去除动作
                    from monai.transforms import RemoveSmallObjects
                    remover = RemoveSmallObjects(min_size=min_voxel_count)
                    val_outputs_list = [remover(p) for p in val_outputs_list]

                    # 2. 计算清理后的指标
                    dice_metric(y_pred=val_outputs_list, y=val_labels_list)


                    val_outputs_post_stack = torch.stack(val_outputs_list, dim=0)
                    val_labels_post_stack = torch.stack(val_labels_list, dim=0)

                    iou_per_class = compute_iou(y_pred=val_outputs_post_stack, y=val_labels_post_stack,
                                                include_background=True)
                    total_iou += iou_per_class.mean().item()

                    # 连通域指标计算
                    for batch_idx in range(val_inputs.shape[0]):
                        if batch_idx < len(val_outputs_list) and batch_idx < len(val_labels_list):
                            idx = 1 if val_outputs_list[batch_idx].shape[0] > 1 else 0
                            val_pred_np_batch = val_outputs_list[batch_idx][idx].cpu().numpy()
                            val_label_np_batch = val_labels_list[batch_idx][idx].cpu().numpy()

                            recall, fpr, tp, fp, fn = calculate_connected_components_metrics(val_pred_np_batch,
                                                                                             val_label_np_batch)
                            total_recall += recall
                            total_fpr += fpr
                            total_tp += tp
                            total_fp += fp
                            total_fn += fn
                            val_samples += 1

                    # 图像保存 (保存过滤后的结果)
                    if args.save_val_images:
                        epoch_val_dir = os.path.join(val_output_dir, f"epoch_{epoch + 1:03d}")
                        os.makedirs(epoch_val_dir, exist_ok=True)

                        for batch_idx in range(len(val_outputs_list)):
                            if "image_meta_dict" in val_data:
                                image_path = val_data["image_meta_dict"]["filename_or_obj"][batch_idx]
                                base_name = os.path.basename(image_path).replace('.nii.gz', '')
                                # 确保 affine 也是 numpy 格式
                                affine = val_data["image_meta_dict"]["affine"][batch_idx]
                                if torch.is_tensor(affine):
                                    affine = affine.cpu().numpy()
                            else:
                                base_name = f"val_{i:03d}_sample_{batch_idx:02d}"
                                affine = np.eye(4)

                            current_pred = val_outputs_list[batch_idx]
                            if current_pred.shape[0] > 1:
                                final_pred_np = torch.argmax(current_pred, dim=0).cpu().numpy().astype(np.uint8)
                            else:
                                final_pred_np = (current_pred[0] > 0.5).cpu().numpy().astype(np.uint8)

                            input_np = val_inputs.cpu().numpy()[batch_idx, 0]
                            label_np = val_labels.cpu().numpy()[batch_idx, 0]

                            save_nifti(input_np, affine,
                                        os.path.join(epoch_val_dir, f"{base_name}_input.nii.gz"),
                                        dtype=np.float32)
                            save_nifti(label_np, affine,
                                        os.path.join(epoch_val_dir, f"{base_name}_label.nii.gz"),
                                        dtype=np.uint8)
                            save_nifti(final_pred_np, affine,
                                        os.path.join(epoch_val_dir, f"{base_name}_pred_filtered.nii.gz"),
                                        dtype=np.uint8)

            # --- 最终汇总 ---
                mean_dice = dice_metric.aggregate().item() if val_samples > 0 else 0.0
                dice_metric.reset()

                mean_iou = total_iou / len(val_loader) if len(val_loader) > 0 else 0.0
                mean_recall = total_recall / val_samples if val_samples > 0 else 0.0
                mean_fpr = total_fpr / val_samples if val_samples > 0 else 0.0
                avg_time_per_volume = total_inference_time / val_samples if val_samples > 0 else 0.0
                val_duration = (datetime.now() - val_start_time).total_seconds()

                logging.info(
                    f"验证完成 | Dice: {mean_dice:.4f}  | IoU: {mean_iou:.4f} | "
                    f"Recall: {mean_recall:.4f} | FPR: {mean_fpr:.4f}")
                logging.info(f"速度性能 | Avg Inference Time: {avg_time_per_volume:.3f}s / volume")
                logging.info(f"详细统计 - TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

                current_lr = optimizer.param_groups[0]['lr']
                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_epoch = epoch + 1
                    torch.save({"model": model.state_dict()}, f"best_metric_model_fold{args.fold}.pth")
                    logging.info(f"最佳指标提升至 {best_metric:.4f}, 模型已保存")

        scheduler.step()
        logging.debug(f"CosineLR调度step: epoch={epoch + 1}")

    final_model_path = f"final_model_fold{args.fold}_epoch{args.epochs}.pth"
    torch.save({
        "model": model.state_dict()
    }, final_model_path)
    logging.info(f"最终模型保存至 {final_model_path}")

    total_duration = (datetime.now() - start_time).total_seconds()
    logging.info(f"训练完成，总耗时: {total_duration:.1f}秒 ({total_duration / 60:.1f}分钟)")
    logging.info(f"最佳 Dice 指标: {best_metric:.4f} (第 {best_epoch} 轮)")

    print(f"训练完成。最佳指标: {best_metric:.4f} 在第 {best_epoch} 轮")
    print(f"详细日志请查看: {log_file}")


def get_args():
    parser = argparse.ArgumentParser(description="Aneurysm Segmentation Training")
    parser.add_argument("--data_dir", type=str, default='../data/IA')
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--crop_size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_val_images", action="store_true", default=True)
    parser.add_argument("--val_output_dir", type=str, default="../val_outputs")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train()