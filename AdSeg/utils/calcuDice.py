import pandas as pd
import GeodisTK
import numpy as np
from scipy import ndimage
import os
import nibabel as nib
import SimpleITK as sitk

"""
calculate the Dice score of two N-d volumes.
s: the segmentation volume of numpy array
g: the ground truth volume of numpy array
"""


def binary_iou(s, g):
    assert (len(s.shape) == len(g.shape))
    # 两者相乘值为1的部分为交集
    intersecion = np.multiply(s, g)
    # 两者相加，值大于0的部分为交集
    union = np.asarray(s + g > 0, np.float32)
    iou = intersecion.sum() / (union.sum() + 1e-10)
    return iou


def binary_dice(s, g):
    assert (len(s.shape) == len(g.shape))
    prod = np.multiply(s, g)
    s0 = prod.sum()
    dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)
    return dice


def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_hausdorff95(s, g, spacing=None):
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if spacing is None:
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if image_dim == 2:
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif image_dim == 3:
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)


def binary_assd(s, g, spacing=None):
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if spacing is None:
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    img = np.zeros_like(s)
    if image_dim == 2:
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif image_dim == 3:
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


def binary_relative_volume_error(s, g):
    s_v = float(s.sum())
    g_v = float(g.sum())
    assert (g_v > 0)
    rve = abs(s_v - g_v) / g_v
    return rve


def compute_class_sens_spec(s, g):
    tp = np.sum((s == 1) & (g == 1))
    tn = np.sum((s == 0) & (g == 0))
    fp = np.sum((s == 1) & (g == 0))
    fn = np.sum((s == 0) & (g == 1))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


def get_evaluation_score(s_volume, g_volume, spacing, metric):
    if len(s_volume.shape) == 4:
        assert (s_volume.shape[0] == 1 and g_volume.shape[0] == 1)
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    if s_volume.shape[0] == 1:
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    metric_lower = metric.lower()

    if metric_lower == "dice":
        score = binary_dice(s_volume, g_volume)

    elif metric_lower == "iou":
        score = binary_iou(s_volume, g_volume)

    elif metric_lower == 'assd':
        score = binary_assd(s_volume, g_volume, spacing)

    elif metric_lower == "hausdorff95":
        score = binary_hausdorff95(s_volume, g_volume, spacing)

    elif metric_lower == "rve":
        score = binary_relative_volume_error(s_volume, g_volume)

    elif metric_lower == "volume":
        voxel_size = 1.0
        for dim in range(len(spacing)):
            voxel_size = voxel_size * spacing[dim]
        score = g_volume.sum() * voxel_size
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))

    return score


def computing_matrix(pred_path, mask_path, save_path):
    seg = sorted(os.listdir(pred_path))
    case_name = []
    ious = []
    dices = []
    hds = []
    rves = []
    asdds = []
    senss = []
    specs = []
    for name in seg:
        if not name.startswith('.') and name.endswith('nii.gz'):
            # 加载label and segmentation image
            seg_ = nib.load(os.path.join(pred_path, name))
            seg_arr = seg_.get_fdata().astype('float32')
            gd_ = nib.load(os.path.join(mask_path, name))
            gd_arr = gd_.get_fdata().astype('float32')
            file = sitk.ReadImage(os.path.join(pred_path, name))
            spacing = file.GetSpacing()
            print(name)
            print(spacing)
            case_name.append(name)
            # 求iou
            iou_score = get_evaluation_score(seg_arr, gd_arr, spacing=spacing, metric='iou')
            ious.append(iou_score)
            # 求hausdorff95距离
            hd_score = get_evaluation_score(seg_arr, gd_arr, spacing=spacing, metric='hausdorff95')
            hds.append(hd_score)
            # 求平均表面距离
            asdd = get_evaluation_score(seg_arr, gd_arr, spacing=spacing, metric='assd')
            asdds.append(asdd)
            # 求体积相关误差
            rve = get_evaluation_score(seg_arr, gd_arr, spacing=spacing, metric='rve')
            rves.append(rve)

            # 求dice
            dice = get_evaluation_score(seg_.get_fdata(), gd_.get_fdata(), spacing=spacing, metric='dice')
            dices.append(dice)
            # 敏感度，特异性
            sens, spec = compute_class_sens_spec(seg_.get_fdata(), gd_.get_fdata())
            senss.append(sens)
            specs.append(spec)
    data = {'iou': ious, 'dice': dices, 'RVE': rves, 'Sens': senss, 'Spec': specs, 'HD95': hds, 'asdd': asdds}
    df = pd.DataFrame(data=data, columns=['iou', 'dice', 'RVE', 'Sens', 'Spec', 'HD95', 'asdd'], index=case_name)
    df.to_excel(os.path.join(save_path, 'Matrix.xlsx'))


if __name__ == '__main__':
    pred_path = r'Q:\Dice2\Dice=0.86\segs\right\masks'
    mask_path = r'Q:\manual\segs\right\masks'
    save_path = r'Q:\Dice2\Dice=0.86\segs\right'
    computing_matrix(pred_path, mask_path, save_path)
