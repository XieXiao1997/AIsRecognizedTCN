import os
import nibabel as nib
import shutil


def repair(root_path):
    for file in os.listdir(root_path):
        file_path = os.path.join(root_path, file)
        img = nib.load(file_path)
        qform = img.get_qform()
        img.set_qform(qform)
        sfrom = img.get_sform()
        img.set_sform(sfrom)
        nib.save(img, file_path)


def movefile(root_path):
    for root, dirs, files in os.walk(root_path):
        for dir in dirs:
            dir_path = os.path.join(root_path, dir)
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                new_path = os.path.join(root_path, file)
                shutil.move(file_path, new_path)
            os.rmdir(dir_path)
