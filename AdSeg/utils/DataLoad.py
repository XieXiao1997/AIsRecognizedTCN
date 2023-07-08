from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandAffined,
    RandFlipd,
    RandShiftIntensityd,
    SpatialPadd,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
    ToTensord,
)
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import numpy as np


def Data_Loader(files, data='train'):
    train_transforms = Compose([
        # 加载图像
        LoadImaged(keys=['image', 'label'])
        # 保证图形为（1,512,512,n)
        , EnsureChannelFirstd(keys=['image', 'label'])
        # 移除图像外边界，集中注意力于图像有效区域
        , CropForegroundd(keys=["image", "label"], source_key="image")
        # 轴坐标重定位 PLI-> represents 3D orientation: (Left, Right), (Posterior, Anterior), (Inferior, Superior).
        , Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 3.0), mode=("bilinear", "nearest"))
        # 图像窗宽窗位重设置
        , Orientationd(keys=['image', 'label'], axcodes='PLI')
        # 图像重采样
        , ScaleIntensityRanged(keys=['image', ], a_min=-110, a_max=190, b_min=0.0, b_max=1.0, clip=True)
        # 添加随机噪声
        , RandGaussianNoised(keys=['image'], prob=0.1)
        # 将所有图像填充至指定大小(-1,)代表自适应填充，填充方式
        , SpatialPadd(keys=['image', 'label'], spatial_size=(-1, -1, 96), mode='empty')
        # 将子图像进行图像增强
        , RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0,
            rotate_range=(0, 0, np.pi / 15),
            scale_range=(0.1, 0.1, 0.1))
        # 从label_key中mask为核心区域进行裁剪，阳性/阴性比为pos/neg，裁剪出num_sample个大小为spatial_size的子图像，
        , RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        )
        # 保证数据格式为Tensor或npArray
        , EnsureTyped(keys=["image", "label"])
    ])
    val_transforms = Compose([
        LoadImaged(keys=['image', 'label'])
        , EnsureChannelFirstd(keys=['image', 'label'])
        , Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 3.0), mode=("bilinear", "nearest"))
        , Orientationd(keys=['image', 'label'], axcodes='PLI')
        , ScaleIntensityRanged(keys=['image'], a_min=-110, a_max=190, b_min=0.0, b_max=1.0, clip=True)
        , CropForegroundd(keys=["image", "label"], source_key="image")
        , EnsureTyped(keys=["image", "label"])
    ])
    if data == 'train':
        ds = Dataset(data=files, transform=train_transforms)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    else:
        ds = Dataset(data=files, transform=val_transforms)
        loader = DataLoader(ds, batch_size=1, num_workers=4)
    lenth = len(ds)
    return loader, lenth
