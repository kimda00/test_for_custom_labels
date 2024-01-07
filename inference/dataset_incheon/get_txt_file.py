import os

TEXT_EXT = [".txt"]
IMAGE_EXT = [".jpg"]

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

if __name__ == '__main__':
    image_path = './val/images'

    imagese = get_image_list(image_path)
    for imgs in imagese:
        imgs = imgs.replace("\\", "/")
        with open("val.txt", "a") as f:
            f.writelines(imgs + '\n')
        

    # with open("val_file_list.txt", "r") as f:
    #     lines = f.readlines()

    # new_lines = ["./images/val/" + line.strip() + "\n" for line in lines]

    # with open("new_file_list.txt", "w") as f:
    #     f.writelines(new_lines)