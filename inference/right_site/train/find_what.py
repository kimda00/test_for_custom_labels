import os

# 이미지 폴더와 라벨 폴더의 경로 설정
image_folder_path = 'images'  # 이미지 폴더 경로를 여기에 입력하세요
label_folder_path = 'changed_labels'  # 라벨 폴더 경로를 여기에 입력하세요

# 각 폴더에서 파일 목록 가져오기
image_files = set(os.listdir(image_folder_path))
label_files = set(f.replace('.txt', '.jpg') for f in os.listdir(label_folder_path))

# 라벨 파일이 없는 이미지 파일 찾기
images_without_labels = image_files - label_files

# 결과 출력
print("라벨이 없는 이미지 파일:")
for image in images_without_labels:
    print(image)
