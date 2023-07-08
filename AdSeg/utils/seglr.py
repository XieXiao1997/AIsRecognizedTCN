import os
import SimpleITK as sitk


def load_data(image_path, label_path):
    img = sitk.ReadImage(image_path)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    img_data = sitk.GetArrayFromImage(img)
    mask = sitk.ReadImage(label_path)
    mask_spacing = mask.GetSpacing()
    mask_origin = mask.GetOrigin()
    mask_direction = mask.GetDirection()
    mask_data = sitk.GetArrayFromImage(mask)
    return img_data, mask_data, (spacing, origin, direction), (mask_spacing, mask_origin, mask_direction)


def segimage_masks(image, mask):
    image_left = image[:, :, 256:]
    image_right = image[:, :, :256]
    mask_left = mask[:, :, 256:]
    mask_right = mask[:, :, :256]
    return image_left, image_right, mask_left, mask_right


def set_data(array, affine):
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(affine[0])
    img.SetOrigin(affine[1])
    img.SetDirection(affine[2])
    return img


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_nii(output_path, image_left_array, image_right_array, label_left_array, label_right_array, name, affine1,
             affine2):
    # 分别保存切割后的文件至指定文件路径
    name = name.split('.')[0]
    img_left = set_data(image_left_array, affine1)
    img_right = set_data(image_right_array, affine1)
    mask_left = set_data(label_left_array, affine2)
    mask_right = set_data(label_right_array, affine2)
    make_dir(os.path.join(output_path, 'left', 'images'))
    make_dir(os.path.join(output_path, 'right', 'images'))
    make_dir(os.path.join(output_path, 'left', 'masks'))
    make_dir(os.path.join(output_path, 'right', 'masks'))
    sitk.WriteImage(img_left, os.path.join(output_path, 'left', 'images', name + '.nii.gz'), useCompression=True)
    sitk.WriteImage(img_right, os.path.join(output_path, 'right', 'images', name + '.nii.gz'), useCompression=True)
    sitk.WriteImage(mask_left, os.path.join(output_path, 'left', 'masks', name + '.nii.gz'), useCompression=True)
    sitk.WriteImage(mask_right, os.path.join(output_path, 'right', 'masks', name + '.nii.gz'), useCompression=True)
    print('{} have saved!'.format(name))


def seg_data(root_path, save_path, mode='train'):
    if mode == 'train':
        labels_path = os.path.join(root_path, 'masks')
    else:
        labels_path = os.path.join(root_path, 'preds')
    images_path = os.path.join(root_path, 'images')
    for file in os.listdir(images_path):
        label_path = os.path.join(labels_path, file)
        image_path = os.path.join(images_path, file)
        image, label, affine1, affine2 = load_data(image_path, label_path)
        image_left_array, image_right_array, label_left_array, label_right_array \
            = segimage_masks(image, label)
        save_nii(save_path, image_left_array, image_right_array, label_left_array, label_right_array, file, affine1,
                 affine2)


if __name__ == '__main__':
    root_path = r'Q:\manual'
    save_path = r'Q:\manual\segs'
    seg_data(root_path, save_path,mode='train')
