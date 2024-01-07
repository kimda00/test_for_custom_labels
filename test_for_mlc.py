import argparse
import yaml
import os
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from utils.general import box_iou, xywh2xyxy
from utils.metrics_for_mlc import ap_per_class, ConfusionMatrix

def load_predicted_labels(pred_label_path):
    labels = {}
    for file in os.listdir(pred_label_path):
        if file.endswith('.txt'):
            with open(os.path.join(pred_label_path, file), 'r') as f:
                image_id = Path(file).stem
                labels[image_id] = [[float(num) for num in x.split()] for x in f.read().strip().splitlines()]
    return labels

def load_gt_labels(gt_label_path):
    labels = {}
    for file in os.listdir(gt_label_path):
        if file.endswith('.txt'):
            with open(os.path.join(gt_label_path, file), 'r') as f:
                image_id = Path(file).stem
                labels[image_id] = [[float(num) for num in x.split()] for x in f.read().strip().splitlines()]
    return labels

def get_class_names(labels_idx):
    with open(labels_idx, 'r') as file:
        class_names = [line.strip() for line in file.readlines()]
        print(f'class name : {class_names}')
    return class_names

def test_with_custom_predictions(gt_label_dir, pred_label_dir, labels_idx, image_dir, output_dir):
    gt_labels = load_gt_labels(gt_label_dir)
    predicted_labels = load_predicted_labels(pred_label_dir)

     # 성능 평가 준비
    class_names = get_class_names(labels_idx)
    nc = len(class_names)  # 클래스 수
    iouv = np.linspace(0.5, 0.95, 10)  # IOU 범위
    niou = len(iouv)
    stats = []  # 통계 데이터 저장
   
    seen = 0

    for image_id, gt_label in tqdm(gt_labels.items()):
        pred_label = predicted_labels.get(image_id, [])
        seen += 1

        if len(pred_label) == 0:
            continue

        pred_label_array = np.array(pred_label)
        if pred_label_array.ndim == 1:
            pred_label_array = np.expand_dims(pred_label_array, axis=0)

        # GT와 예측된 라벨의 비교
        tbox_numpy = xywh2xyxy(np.array([x[1:] for x in gt_label]))
        tbox = torch.tensor(tbox_numpy) # GT 박스
        tcls = torch.tensor([x[0] for x in gt_label]) # GT 클래스
        pbox_numpy = xywh2xyxy(np.array([x[1:] for x in pred_label])) 
        pbox = torch.tensor(pbox_numpy)   # 예측된 박스
        pcls = torch.tensor([x[0] for x in pred_label])  # 예측된 클래스
        # scores = np.array([x[5] for x in pred_label])  # 예측된 점수

        # IOU 계산
        iou = box_iou(pbox, tbox) #각 예측 박스와 모든 실제 박스 간의 IOU 값
        iou_maxes = iou.max(dim=1).values

        iouv = torch.linspace(0.5, 0.95, 10)  # IOU 범위 (PyTorch Tensor로 변환)
        
        # 각 iou 값이 iouv 범위 내에 있는지 확인
        correct = iou_maxes[:, None] > iouv[None, :]  # 크기 조정 및 비교

        x = [iou.max(1), iou.max(0)]  # 최대 IOU 값
     
        # 성능 평가 지표 계산
        conf = [1.0 for _ in range(len(pcls))] 
        """
        TODO : add confidence score on pred labels, and make gt label's confidence 1
        """
        for ciou in correct:
            stats.append((ciou, torch.tensor(conf), pcls, tcls))
          
        for image_id, gt_label in gt_labels.items():
            image_path = Path(image_dir) / (image_id + '.jpg')  # 원본 이미지 경로
            output_path = Path(output_dir) / (image_id + '.jpg')  # 출력 이미지 경로

            gt_boxes = []
            for x in gt_label:
                box = np.array(x[1:])
                box_converted = xywh2xyxy(box.reshape(1, -1))[0] 
                gt_boxes.append(box_converted)

            pred_boxes = []
            for x in predicted_labels.get(image_id, []):
                box = np.array(x[1:])
                box_converted = xywh2xyxy(box.reshape(1, -1))[0] 
                pred_boxes.append(box_converted)
            # draw_boxes(image_path, gt_boxes, pred_boxes, output_path)

    # 성능 평가
    print("성능 평가 시작")
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
    print("성능 평가 완료")
  
    # 전체 데이터셋에 대한 결과 출력
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # 클래스별 타겟 수

        # 전체 및 클래스별 결과 출력
        print('Class    Images     Labels          P          R     mAP@.5 mAP@.5:.95')
        for i, c in enumerate(ap_class):
            class_name = class_names[c] if c < len(class_names) else 'unknown'
            print(f'{class_name:>10}...')
            print(f'{class_name[c]:>10}{seen:>10}{nt[c]:>10}{p[i]:>10.3f}{r[i]:>10.3f}{ap50[i]:>10.3f}{ap[i]:>10.3f}')
        print(f'{"all":>10}{seen:>10}{nt.sum():>10}{mp:>10.3f}{mr:>10.3f}{map50:>10.3f}{map:>10.3f}')
    
    # 혼동 행렬
        """
        TODO : MAKE CONFUSION MATRIX 
        """
    # confusion_matrix.plot(save_dir=None, names=data['names'])
        
def draw_boxes(image_path, gt_boxes, pred_boxes, output_path):
    # 이미지 로드
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    width, height = image.size
    def scale_box(box):
        x_min, y_min, x_max, y_max = box
        return [
            x_min * width, y_min * height,
            x_max * width, y_max * height
        ]

    # GT 박스 그리기 (초록색)
    for box in gt_boxes:
        draw.rectangle(scale_box(box), outline="green", width=2)

    # 예측된 박스 그리기 (빨간색)
    for box in pred_boxes:
        draw.rectangle(scale_box(box), outline="red", width=2)

    # 이미지 저장
    image.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    # parser.add_argument('--data', type=str, default='data/etri.yaml', help='*.yaml path')
    parser.add_argument('--pred-label-dir', type=str, default='inference/pred_labels_sample/labels', help='Path to predicted labels')
    parser.add_argument('--gt-label-dir', type=str, default='inference/right_site/val/labels', help='Path to predicted labels')
    parser.add_argument('--labels_idx', type=str, default='label_idx/dataset_incheon.labels', help='Path to predicted labels')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--image-dir', type=str, default='inference/right_site/val/images', help='Path to images')
    parser.add_argument('--output-dir', type=str, default='check_box', help='check_box')
    # parser.add_argument('--img', type=int, default=640, help='inference size (pixels)')

    opt = parser.parse_args()

    test_with_custom_predictions(opt.gt_label_dir, opt.pred_label_dir, opt.labels_idx, opt.image_dir, opt.output_dir)