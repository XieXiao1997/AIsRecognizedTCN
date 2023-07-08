import os
import SimpleITK as sitk
import heapq as hp


def post_processing(path):
    mask = sitk.ReadImage(path, sitk.sitkUInt16)
    mask = sitk.BinaryFillhole(mask)
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, mask)
    size = []
    for l in stats.GetLabels():
        size.append(stats.GetPhysicalSize(l))
    max2 = list(map(size.index, hp.nlargest(2, size)))
    max2 = [x + 1 for x in max2]
    labelmaskimage = sitk.GetArrayFromImage(cc)
    outmask = labelmaskimage.copy()
    for i in stats.GetLabels():
        if i in max2:
            outmask[labelmaskimage == i] = 1
        else:
            outmask[labelmaskimage == i] = 0
    outmask = outmask.astype('float32')
    out = sitk.GetImageFromArray(outmask)
    out.SetDirection(mask.GetDirection())
    out.SetSpacing(mask.GetSpacing())
    out.SetOrigin(mask.GetOrigin())
    return out


def pp(root_path, save_path=None):
    for file in os.listdir(root_path):
        file_path = os.path.join(root_path, file)
        out_file = post_processing(file_path)
        print(f"{file} has saved!")
        if save_path != None:
            sitk.WriteImage(out_file, os.path.join(save_path,file))
        else:
            sitk.WriteImage(out_file, file_path)


if __name__ == '__main__':
    root_path = r'Q:\Dice\pre-preds'
    save_path = r'Q:\Dice\post-preds'
    pp(root_path, save_path=save_path)
