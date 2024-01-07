import cv2
import shutil
import os
import matplotlib.pyplot as plt
from random import randint
import numpy as np

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
txt_EXT = [".txt"]
# epiton_label_list = [ 'person', 'bicycle','car','bus', 'motorcycle','truck', 'tl_v', 'tl_p', 'traffic_sign', 'traffic_light']
epiton_label_list = [ 'person', 'bicycle','car','bus', 'motorcycle','truck', 'green', 'red', 'yellow', 'red_arrow', 'red_yellow','green_arrow','green_yellow','green_right','warn','black','tl_v', 'tl_p', 'traffic_sign', 'traffic_light']

                          #0        1       2     3             4         5       6         7         8         9             10         11           12           13                14             15         16         17  
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
      print(maindir)
      for filename in file_name_list:
          apath = os.path.join(maindir, filename)
          ext = os.path.splitext(apath)[1]
          if ext in IMAGE_EXT:
              image_names.append(apath)
    return image_names

def get_txt_list(path):
    txt_names = []
    for maindir, subdir, file_name_list in os.walk(path):
      print(maindir)
      for filename in file_name_list:
          apath = os.path.join(maindir, filename)
          ext = os.path.splitext(apath)[1]
          if ext in txt_EXT:
              txt_names.append(apath)
    return txt_names

def color_random():
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    return rand_color_list

def make_dir(dir):
    if(not os.path.exists(dir)):
        os.makedirs(dir)

class divid:
  def __init__(self):
    self.save_dir = './pviz/'
    self.img_size = 0,0
    self.color_list = color_random()

  def get_bbox(self,txt):
    data_a = []
    with open(txt,'r')as f:
      data = f.readlines()
      for da in data:
        label = int(da.split()[0])
        box = [round(float(x),3) for x in da.split()[1:]]
        box.insert(0,label)
        data_a.append(box)
    bboxes =self.bbox_pro(data_a)
    return bboxes

  def bbox_pro(self,bboxes):
    bbox_list = []
    for bbox in bboxes:
      label,xmid,ymid,bbox_w,bbox_h =bbox[0], bbox[1]*self.img_size[0],bbox[2]*self.img_size[1],bbox[3]*self.img_size[0],bbox[4]*self.img_size[1]
      xmax = xmid+bbox_w//2
      ymax = ymid+bbox_h//2
      xmin = xmid-bbox_w//2
      ymin = ymid-bbox_h//2
      bbox_list.append([label,xmin,ymin,xmax,ymax])
    return bbox_list

  def viz(self,img_path,txt,save_dir):
    fname = img_path.split('/')[-1]
    img = cv2.imread(img_path) #Read the img
    self.img_size = (img.shape[1],img.shape[0])
    bboxes = self.get_bbox(txt)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tl = 1 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)  # font thickness
    for bb in bboxes:
      c1, c2 = (int(bb[1]), int(bb[2])), (int(bb[3]), int(bb[4]))
      cv2.rectangle(img, c1, c2, self.color_list[int(bb[0])], thickness=tl, lineType=cv2.LINE_AA)
      tf = max(tl - 1, 1)  # font thickness
      t_size = cv2.getTextSize(epiton_label_list[int(bb[0])], 0, fontScale=tl / 3, thickness=tf)[0]
      c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
      cv2.rectangle(img, c1, c2, self.color_list[int(bb[0])], -1, cv2.LINE_AA)  # filled
      cv2.putText(img, epiton_label_list[int(bb[0])], (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    cv2.imwrite(save_dir + '/' + fname,img)


  def main(self,img,txt):
    for index,image in enumerate(img):
      xx_txt = txt[index]
      viz_save_dir = self.save_dir
      make_dir(viz_save_dir)
      ### do process with each file
      self.viz(image,xx_txt,viz_save_dir)


if __name__ == '__main__':
  divid = divid()
  img_path = 'images'
  txt_path = 'changed_labels'
  img = sorted(get_image_list(img_path))
  txt = sorted(get_txt_list(txt_path))
  divid.main(img,txt)