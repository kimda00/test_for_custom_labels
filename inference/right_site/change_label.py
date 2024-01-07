import os
import yaml
import tqdm

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


def get_text_list(path):
    text_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in TEXT_EXT:
                text_names.append(apath)
        
    return text_names

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

if __name__ == '__main__':
    
    make_dir('./val/changed_labels')
   
    
    # darknet파일 리스트화 시킴
    with open('dataset_incheon.labels', 'r') as darknet_label:
        new_label_list = [line.strip() for line in darknet_label.readlines()]
    #print("이걸로 변경")
    #print(change_label_list)
    
    # 원래 리스트 불러오기
    yaml_file_path = "./data.yaml"
    with open(yaml_file_path, "r") as y:
        yaml_dict = yaml.safe_load(y)
        origin_list = yaml_dict["names"]
    #print("원래꺼")
    
    # 원래꺼 인덱스와
    for i, value in enumerate(origin_list):    
        index = origin_list.index(value)
        #print(f"Index: {index}, Value: {value}")

    # 새로운거 인덱스 화
    for i, value in enumerate(new_label_list):    
        index = new_label_list.index(value)
        #print(f"Index: {index}, Value: {value}")

    # 원래파일 쭉 읽음
    label_path = './val/labels'
    label_list = get_text_list(label_path)
    # 원래 리스트와 새 리스트 사이의 매핑 생성
    label_mapping = {}
    for old_label in origin_list:
        if old_label in new_label_list:
            label_mapping[origin_list.index(old_label)] = new_label_list.index(old_label)


    for file_path in tqdm.tqdm(label_list):
        with open(file_path,'r') as f:
            labels = f.readlines()
        file_name = file_path[15:-3]
        # print(file_name)
        folder_name = "changed_labels"
        # os.makedirs(folder_name, exist_ok=True)
        folder_path = "./val/"+folder_name
            #print(labels)
        for t in labels:
            split_labels_list = t.split()
            # print(t)
            #rint(split_labels_list[0], origin_list[int(split_labels_list[0])])
            sd = int(split_labels_list[0])
            try:
                a = new_label_list.index(origin_list[sd])
            except:
                print(file_path)
                print(t)
                break
            split_labels_list[0] = str(a)
            with open(f'{os.path.join(folder_path, file_name)}txt', 'a') as n:
                c = 0
                for b in split_labels_list:
                    c += 1
                    n.writelines(b)
                    n.write(' ')
                    if c%5 ==0 :
                        n.write('\n')