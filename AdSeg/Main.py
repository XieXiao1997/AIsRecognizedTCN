import os
from inference import pred_img_by_model
from utils.utils import repair, movefile
from utils.seglr import seg_data
from utils.post_processing import pp


def img_pred(root_path, model_path):
    img_path = os.path.join(root_path, 'images')
    out_path = os.path.join(root_path, 'preds')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    pred_img_by_model(img_path, out_path, model_path)
    movefile(out_path)
    repair(out_path)
    pp(out_path)


def seg_left_right(root_path, mode='Train'):
    out_path = os.path.join(root_path, 'segs')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    seg_data(root_path, out_path, mode=mode)


if __name__ == '__main__':
    root_path = r'Q:\extest3'
    model_path = r'C:\Users\Administrator\PycharmProjects\AS\model_save\res=2\182best_model_dice_v0.8583t0.8316.pth'
    img_pred(root_path, model_path)
    seg_left_right(root_path, mode='F')
