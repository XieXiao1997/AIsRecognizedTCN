import pandas as pd
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
)
from monai.networks.nets import UNet, UNETR
from monai.networks.layers import Norm, Act
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.losses import DiceLoss, DiceFocalLoss, MaskedDiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from utils.DataLoad import Data_Loader
import torch
import os
import glob
import time


set_determinism(seed=0)
device = torch.device("cuda:0")
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True


def train(train_files, val_files):
    train_loader, train_lenth = Data_Loader(train_files, data='train')
    val_loader, val_lenth = Data_Loader(val_files, data='val')
    model = UNet(dimensions=3
                 , in_channels=1
                 , out_channels=2
                 , channels=(32, 64, 128, 256, 512)
                 , strides=(2, 2, 2, 2)
                 , num_res_units=0
                 # , dropout=0.5
                 , norm=Norm.BATCH).to(device)
    # model.load_state_dict(torch.load('pretrain_model/Unet.pth'), strict=False)
    # 定义损失函数
    # loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True)
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    # DiceMetric:比较两特征张量之间的差别
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    # hausdorff = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95,
    #                                     reduction='mean', )
    # surface = SurfaceDistanceMetric(include_background=False, distance_metric='euclidean', reduction='mean')
    # 定义学习率下降策略
    scheduler = StepLR(optimizer, step_size=20, gamma=0.90)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, verbose=True)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)
    #
    # 训练参数
    max_epochs = 200
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    metric_values = []
    step_total = 0
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, num_classes=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=2)])
    # 训练日志保存
    log_dir = 'LOG/res=2256'
    writer = SummaryWriter(log_dir)

    # 开始训练
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            epoch_start = time.time()
            step_start = time.time()
            step_total += 1
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            train_outputs = [post_pred(i) for i in decollate_batch(outputs)]
            train_labels = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=train_outputs, y=train_labels)
            # hausdorff(y_pred=train_outputs, y=train_labels)
            # surface(y_pred=train_outputs, y=train_labels)
            print(dice_metric.aggregate().item())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            writer.add_scalar('Training/Step loss', loss.item(), step_total + 1)
            print('{}/{}, train_loss: {:.4f}, step time: {:.4f} '.format(
                step, train_lenth // train_loader.batch_size, loss.item(), time.time() - step_start))
        train_metric = dice_metric.aggregate().item()
        # train_hs = hausdorff.aggregate().item()
        # train_surface = surface.aggregate().item()
        writer.add_scalar('Training/Dice metric', train_metric, epoch + 1)
        # writer.add_scalar('Training/Hausdorff', train_hs, epoch + 1)
        # writer.add_scalar('Training/Surface', train_surface, epoch + 1)
        print(train_metric)
        print('**************************')
        # print(train_hs)
        print('**************************')
        # print(train_surface)
        dice_metric.reset()
        # hausdorff.reset()
        # surface.reset()
        lr_rate = scheduler.get_last_lr()[0]
        writer.add_scalar('Training/Learning Rate', lr_rate, epoch + 1)
        print(lr_rate)
        scheduler.step()
        epoch_loss /= step
        writer.add_scalar("Training/Epoch Average Loss", epoch_loss, epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    # print(val_inputs.shape,val_labels.shape)
                    roi_size = (96, 96, 96)
                    sw_batch_size = 8
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=0.5)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # 计算当前batch的metric
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    # hausdorff(y_pred=val_outputs, y=val_labels)
                    # surface(y_pred=val_outputs, y=val_labels)
                    print(dice_metric.aggregate().item())
                # 统计最终metric结果
                val_metric = dice_metric.aggregate().item()
                # val_hs = hausdorff.aggregate().item()
                # val_surface = surface.aggregate().item()
                writer.add_scalar("Validation/Dice metric", val_metric, epoch + 1)
                # writer.add_scalar("Validation/Hausdorff", val_hs, epoch + 1)
                # writer.add_scalar("Validation/Surface", val_surface, epoch + 1)
                # 重置dice矩阵
                dice_metric.reset()
                # hausdorff.reset()
                # surface.reset()
                metric_values.append(val_metric)
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_metric_epoch = epoch + 1
                    save_dir = 'model_save/res=2256/'
                    save_path = save_dir + str(epoch + 1) + "best_model_dice_v{:.4f}t{:.4f}.pth".format(
                        val_metric,
                        train_metric)
                    torch.save(model.state_dict(), save_path)
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {val_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    writer.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 加载训练集数据
    train_root = r'Q:\deep\train'
    train_images = sorted(glob.glob(os.path.join(train_root, 'images', '*.nii')))
    train_labels = sorted(glob.glob(os.path.join(train_root, 'masks', '*.nii.gz')))
    train_files = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(train_images, train_labels)]
    # 加载验证集数据
    valid_root = r'Q:\deep\valid'
    valid_images = sorted(glob.glob(os.path.join(valid_root, 'images', '*.nii')))
    valid_labels = sorted(glob.glob(os.path.join(valid_root, 'masks', '*.nii.gz')))
    valid_files = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(valid_images, valid_labels)]
    train(train_files, valid_files)

    # for item in val_files:
    #     print(item['image'])
    # test(test_files)
