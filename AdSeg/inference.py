from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    Invertd,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm

from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
import torch
import os
import glob
from utils.utils import repair, movefile


def pred_img_by_model(test_file_path, out_path, model_path):
    test_images = sorted(
        glob.glob(os.path.join(test_file_path, "*.nii.gz")))

    test_data = [{"image": image} for image in test_images]
    test_org_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=["image"], axcodes="PLI"),
            Spacingd(keys=["image"], pixdim=(
                1.0, 1.0, 3.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-110, a_max=190,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys="image"),
        ]
    )

    test_org_ds = Dataset(
        data=test_data, transform=test_org_transforms)

    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=test_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=out_path, output_postfix='',
                   resample=False),
    ])

    device = torch.device("cuda:0")
    model = UNet(dimensions=3
                 , in_channels=1
                 , out_channels=2
                 , channels=(32, 64, 128, 256, 512)
                 , strides=(2, 2, 2, 2)
                 , num_res_units=2
                 # , dropout=0.5
                 , norm=Norm.BATCH).to(device)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs = test_data["image"].to(device)
            roi_size = (96, 96, 96)
            sw_batch_size = 8
            test_data["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            print('A person seg save')


if __name__ == "__main__":
    test_file_path = '../Sample'
    out_path = '../Output'
    model_path = './model_save/3DResUnet/182best_model.pth'
    pred_img_by_model(test_file_path, out_path, model_path)
    movefile(out_path)
    repair(out_path)
