import os
val_img_dir = 'data/tinyimagenet/val/images'
if os.path.exists(val_img_dir):
    data = open('data/tinyimagenet/val/val_annotations.txt', 'r').readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]

    for img, folder in val_img_dict.items():
        new_dir = f'data/tinyimagenet/val/{folder}'
        os.makedirs(new_dir, exist_ok=True)
        old_file = f'{val_img_dir}/{img}'
        if os.path.exists(old_file):
            os.rename(old_file, f'{new_dir}/{img}')

    import shutil
    shutil.rmtree(val_img_dir)
