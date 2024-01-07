import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics_for_mlc import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def load_labels(label_path):
    # 각 이미지에 대한 라벨을 저장할 딕셔너리
    labels = {}
    # 라벨 경로에서 모든 .txt 파일을 로드
    for file in os.listdir(label_path):
        if file.endswith('.txt'):
            with open(os.path.join(label_path, file), 'r') as f:
                # 이미지 ID는 파일 이름의 stem 부분입니다 (확장자를 제외한 부분)
                image_id = Path(file).stem
                # 문자열을 공백으로 분할하고, 각 값을 실수로 변환
                labels[image_id] = [[float(num) for num in x.split()] for x in f.read().strip().splitlines()]
                #if len(labels) == 1:
                   # print(f"Loaded labels for {image_id}: {labels[image_id]}")
    
    return labels

# def plot_boxes(image_path, pred_boxes, gt_boxes, save_path):
#     # 이미지 로드
#     image = Image.open(image_path)
#     fig, ax = plt.subplots(1, figsize=(12, 9))
#     ax.imshow(image)

#     # 예측된 바운딩 박스 그리기 (빨간색)
#     for box in pred_boxes:
#         rect = patches.Rectangle((box[1], box[2]), box[3] - box[1], box[4] - box[2], linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)

#     # 실제 바운딩 박스 그리기 (초록색)
#     for box in gt_boxes:
#         rect = patches.Rectangle((box[1], box[2]), box[3] - box[1], box[4] - box[2], linewidth=2, edgecolor='g', facecolor='none')
#         ax.add_patch(rect)

#     # plt.show()/
#     plt.axis('off')  # 축 제거
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()

def plot_boxes(image_path, pred_boxes, gt_boxes, save_path):
    # 이미지 로드
    image = Image.open(image_path)
    # 이미지 크기가 640x640으로 고정되어 있다고 가정
    figsize = (6.4, 6.4)  # 640 / 100 = 6.4

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    # 예측된 바운딩 박스 그리기 (빨간색)
    for box in pred_boxes:
        rect = patches.Rectangle((box[1], box[2]), box[3] - box[1], box[4] - box[2], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # 실제 바운딩 박스 그리기 (초록색)
    for box in gt_boxes:
        rect = patches.Rectangle((box[1], box[2]), box[3] - box[1], box[4] - box[2], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # print(f"Drawing boxes on {image_path}")
    # print(f"Predicted boxes: {pred_boxes}")  # 예측된 바운딩 박스 좌표 출력
    # print(f"Ground truth boxes: {gt_boxes}")

    plt.axis('off')  # 축 제거
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    plt.savefig(save_path, bbox_inches='tight',pad_inches=0)
    plt.close()

def test(opt, data,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,
         single_cls=False,
         augment=False,
         verbose=False,
         dataloader=None,
         save_dir=Path(''),
         save_txt=False,
         save_hybrid=False,
         save_conf=False,
         plots=True,
         wandb_logger=None,
         half_precision=True,
         is_coco=False,
         v5_metric=False):
    
    # Set device
    device = select_device('', batch_size=batch_size)

    # Configure
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    
    # Dataloader
    task = 'val'  # path to validation images
    dataloader = create_dataloader(data[task], imgsz, batch_size, 32, opt=opt, pad=0.5, rect=True,
                                   prefix=colorstr(f'{task}: '))[0] 
    # print(f'imgsz is {imgsz}')
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = data['names']
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    stats = []

    #######################################
    # Load inferred labels
    inferred_labels = load_labels('inference/pred_labels_sample/labels') 

    #######################################

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        # print(f'target : {targets}')
        nb, _, height, width = img.shape  # batch size, channels, height, width
            
        for si, path in enumerate(paths):
            image_id = Path(path).stem

            shape = shapes[si][0]  # 현재 이미지의 원본 크기를 가져옵니다.
         
            orig_w, orig_h = shapes[si][0][0], shapes[si][0][1]  # 원본 너비와 높이

            current_targets = targets[targets[:, 0] == si, :]
            # print(f'original gt : {current_targets}')
            # current_targets = current_targets[:, 1:]
            gt_scale_factor = 640/640
            current_targets[:, 2] *=  gt_scale_factor # x_center 정규화
            current_targets[:, 3] *= gt_scale_factor # y_center 정규화
            current_targets[:, 4] *= gt_scale_factor  # width 정규화
            current_targets[:, 5] *= gt_scale_factor  # height 정규화

            if image_id in inferred_labels:
                predn = torch.tensor([[float(x) for x in det] for det in inferred_labels[image_id]], device=device)
                if predn.shape[1] == 5:  # [class, x_center, y_center, width, height] 형식인 경우
                    # confidence 값이 없으므로, 임시로 1.0을 추가합니다.
                    predn = torch.cat((predn, torch.ones((predn.shape[0], 1), device=predn.device)), dim=1)
               
                if predn.nelement() == 0:  # .nelement() gives the total number of elements in the tensor
                    # print(f"No predictions for image ID {image_id}")
                    continue

                # predn 형식 확인 및 수정
                predn = predn[:, [1, 2, 3, 4, 5, 0]] 
                # print(f'predn : {predn}')
                scale_factor = 672/640
    
                      # # predn 텐서의 각 좌표를 원본 이미지 크기로 나누어 정규화합니다pd.
                predn[:, 0] *= scale_factor  # x_center 정규화
                predn[:, 1] *= scale_factor   # y_center 정규화
                predn[:, 2] *= scale_factor   # width 정규화
                predn[:, 3] *= scale_factor   # height 정규화
             
                predn[:, 0:4] = scale_coords(img[si].shape[1:], predn[:, 0:4], shapes[si][0])
                
                # Compute IoU with GT
                # labels = targets[targets[:, 0] == si, 1:]
                labels = current_targets[current_targets[:, 0] == si, 1:]
                # print(f'labels : {labels}')

                # labels[:, 1:5] = scale_coords(img[si].shape[1:], labels[:, 1:5], shapes[si][0])
                
                tcls = labels[:, 0].tolist() if len(labels) else []  # target class
                p_cls = predn[:, 5].tolist() if len(predn) else [] 
        
                if len(predn):
                    # Statistics per image
                    class_pred = predn[:, 5].unsqueeze(1)
                    class_label = labels[:, 0].unsqueeze(1)
                   
                    predn_xyxy = xywh2xyxy(predn[:,0:4])
                    labels_xyxy = xywh2xyxy(labels[:, 1:5])

                    predn_xyxy_label = torch.cat((class_pred, predn_xyxy), dim=1)
                    gt_xyxy_label = torch.cat((class_label, labels_xyxy), dim=1)
                    # print(f'glxy : {gt_xyxy_label}')
                    
                    predn_xyxy_label[:, [1,2,3,4]] *= 672
                    gt_xyxy_label[:, [1,2,3,4]]*= 640

                    ious, i = box_iou(predn_xyxy_label[:,1:5], gt_xyxy_label[:, 1:5]).max(1)  # best ious and indices
                    # print(f"IoUs for {image_id}:", ious)
                    print(f"IoUs: {ious}")
                    
                    correct = ious > iou_thres
                
                    stats.append((correct.cpu(), predn[:, 4].cpu(), p_cls, tcls))
                    # 클래스 라벨 비교를 위한 검증 코드
                    # for pred_cls, target_cls in zip(p_cls, tcls):
                    #     if pred_cls == target_cls:
                    #         print(f"Correct class prediction for class {pred_cls} in image {image_id}")
                    #     else:
                    #         print(f"Incorrect class prediction: GT {target_cls}, Predicted {pred_cls} in image {image_id}")

    
                seen += 1
                save_path = f'/home/da0/yolov7/check_box/{image_id}.png'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                plot_boxes(
                    image_path=path,  # 이미지 파일 경로
                    pred_boxes=predn_xyxy_label.cpu().numpy(),  # 예측된 바운딩 박스
                    gt_boxes=gt_xyxy_label.cpu().numpy() , # 실제 바운딩 박스
                    save_path = save_path )
    ap_class=[]

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # print("Statistics collected for current image:", stats[-1])
    # stats_np = [stat.cpu().numpy() if torch.is_tensor(stat) else np.array(stat) for stat in stats]
    if len(stats) and stats[0].any():
        # After the loop ends and before calling ap_per_class
        # print("All collected statistics:", stats)
        # print("Stats before ap_per_class() call:", stats)
        # p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap, f1, ap_class = ap_per_class(np.array(stats[0]), np.array(stats[1]), np.array(stats[2]), np.array(stats[3]))
        # print("Stats after ap_per_class() call:", stats)
        # After calling ap_per_class
        print("Precision, Recall, and AP for each class:", p, r, ap)

        mp, mr, map50, map = p.mean(), r.mean(), ap.mean(0), ap.mean(1)
        stats[3] = [np.array([x]) if np.isscalar(x) else x for x in stats[2]]

        nt = np.bincount(np.concatenate(stats[3], axis=0).astype(np.int64), minlength=nc)  # number of targets per class

            # Print results
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        map50 = ap[:, 0].mean() if ap.ndim > 1 else ap.mean()
        map = ap.mean()

        print("nt.sum():", nt.sum())
        print("mp:", mp)
        print("mr:", mr)
        print("map50:", map50)
        print("map:", map)
    
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    else:
        nt = torch.zeros(1)
        mp, mr, map50, map = torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
    
    # # Print results
    # pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    # map50 = ap[:, 0].mean() if ap.ndim > 1 else ap.mean()
    # map = ap.mean()

    # print("nt.sum():", nt.sum())
    # print("mp:", mp)
    # print("mr:", mr)
    # print("map50:", map50)
    # print("map:", map)
   
    # print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # # ... (출력 및 결과 저장 코드 생략) ...

    return (mp, mr, map50, map), ap_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt, opt.data,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             v5_metric=opt.v5_metric
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data,
                 opt.batch_size,
                 opt.img_size,
                 0.25,  # Note: conf_thres 
                 0.45,  # Note: iou_thres .
                 opt.single_cls,
                 opt.augment,
                 opt.verbose,
                 save_txt=False,
                 save_hybrid=False,
                 save_conf=False,
                 plots=False,
                 v5_metric=opt.v5_metric
                 )
    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
