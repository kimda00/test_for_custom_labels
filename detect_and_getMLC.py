import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
os.environ['CUDA_VISIBLE_DEVICES'] = ''
model = load_model('weights/500_model_weights.h5')


def run_testing(cropped_img):
    # model = load_model('weights/500_model_weights.h5')  

    img_width = 150
    img_height = 150

    data = pd.read_csv('500_label_table_1030.csv')
    classes = data.columns[2:]

    img_resized = cv2.resize(cropped_img, (img_width, img_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = image.img_to_array(img_rgb)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, img_width, img_height, 3)
    
    y_prob = model.predict(img_array)
    top3 = np.argsort(y_prob[0])[:-3:-1]
    predictions = [classes[i] for i in top3]
    percentage = [y_prob[0][i] for i in top3]
    
    over_percent = [p for p in percentage if p >= 0.05]

    # percentage가 10% 이상인 예측이 하나만 있고, 그 예측이 'red'인 경우
    if len(over_percent) == 1 and predictions[0] == 'red':
        return 6 , predictions, percentage # 4번 클래스 반환
    if len(over_percent) == 1 and predictions[0] == 'green':
        return 4, predictions, percentage # 4번 클래스 반환
    if len(over_percent) == 1 and predictions[0] == 'yellow':
        return 8, predictions, percentage # 4번 클래스 반환
    if len(over_percent) == 1 and predictions[0] == 'black':
        return 12, predictions, percentage # 4번 클래스 반환

    # percentage가 10% 이상인 예측이 두 개 있고, 가장 높은 확률의 예측이 'red'이며, 다음으로 높은 확률의 예측이 'arrow'인 경우
    if len(over_percent) == 2 and predictions[0] == 'red' and predictions[1] == 'arrow':
        return 9, predictions, percentage  # 12번 클래스 반환
    if len(over_percent) == 2 and predictions[0] == 'red' and predictions[1] == 'yellow':
        return 10, predictions, percentage  # 12번 클래스 반환
    if len(over_percent) == 2 and predictions[0] == 'green' and predictions[1] == 'arrow':
        return 11, predictions, percentage # 12번 클래스 반환

    # 위 조건에 해당하지 않는 경우
    else:
        return -1, predictions, percentage

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            #  # 기존 레이블 파일을 읽어서 리스트로 저장
            # if os.path.isfile(txt_path):
            #     with open(txt_path, 'r') as file:
            #         old_labels = [x.split() for x in file.read().strip().splitlines()]
            # else:
            #     print(f"No label file found for {txt_path}, skipping update.")
            #     continue
            old_labels = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 결과를 처리하는 부분
                for j, (*xyxy, conf, cls) in enumerate(det):
                    old_labels.append([cls, *xyxy, conf])
                    if cls in [4,6,8,9,10,11,12]:

                        # 감지된 객체를 잘라내기
                        crop_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        # 추가 분류를 수행
                        new_cls, predictions, percentage = run_testing(crop_img)
                        
                        # 새로운 클래스가 유효한 경우(-1이 아닌 경우) 해당 레이블의 클래스 번호를 업데이트
                        if new_cls != -1 and new_cls != int(old_labels[j][0]):
                            old_labels[j][0] = str(new_cls)
            # Get image dimensions
            img_width = im0.shape[1]
            img_height = im0.shape[0]

            # Write the updated labels to the file
            if save_txt and old_labels:
                with open(txt_path + '.txt', 'w') as file:
                    for label in old_labels:
                        # Convert Tensor to float if necessary
                        class_id = int(label[0]) if isinstance(label[0], torch.Tensor) else int(label[0])
                        bbox = [x.item() if isinstance(x, torch.Tensor) else x for x in label[1:5]]
                        conf = label[5].item() if isinstance(label[5], torch.Tensor) else label[5]

                        # Convert bbox from absolute to relative format (x_center, y_center, width, height)
                        x_center = ((bbox[0] + bbox[2]) / 2) / img_width
                        y_center = ((bbox[1] + bbox[3]) / 2) / img_height
                        width = (bbox[2] - bbox[0]) / img_width
                        height = (bbox[3] - bbox[1]) / img_height

                        # Write to file
                        file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

     
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

#    if save_txt or save_img:
#        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


