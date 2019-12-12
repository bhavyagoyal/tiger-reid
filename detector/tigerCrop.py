#Detector parsing

import os
import glob
import sys
import cv2
import numpy as np 
import json

BASE_FOLDER = os.path.expanduser("~/datasets/cvwc/")
OUTPUT_FOLDER = BASE_FOLDER + "crops3/"
img_files = glob.glob(BASE_FOLDER + "atrw_detection_test/test/*.jpg")
json_files = glob.glob(BASE_FOLDER + "test_inference3/*.json")
BBOX_FILE = OUTPUT_FOLDER+"bbox.json"
METRIC_FILE = OUTPUT_FOLDER+"metric_detection_test.txt"

# class_labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat","chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor", "Tiger"]
class_labels = ["Tiger"]

# Create image crops
img_files.sort()
json_files.sort()
counter = 0
all_bboxes = []
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

mf = open(METRIC_FILE, 'w')
for i in range(len(img_files)):
    img = cv2.imread(img_files[i])
    with open(json_files[i]) as f:
        data = json.load(f)
        bboxes = data["bboxes"]
        labels = data["pred_classes"]
        scores = data["scores"]
    f.close()

    for j in range(len(bboxes)):
        cat = class_labels[labels[j]]
        if cat != "Tiger" or scores[j]<=0.8:
            continue
#         if cat in ['zebra', 'cat', 'cow', 'horse', 'dog' ] or scores[j]<=0.8:
#             continue
        temp = {}
        counter = counter+1
        temp["bbox_id"] = counter
        temp["image_id"] = int(json_files[i].split("/")[-1].split(".")[0])
        # x, y, w, h
        # x <- xmin
        # y <- ymin
        # w <- xmax -xmin
        # h <- ymax -ymin
        temp["pos"] = [int(bboxes[j][0]), int(bboxes[j][1]), (int(bboxes[j][2]) - int(bboxes[j][0])), (int(bboxes[j][3]) - int(bboxes[j][1]))]
        all_bboxes.append(temp)                
        img_temp = img
#         crop = img_temp[max(0,int(0.95*bboxes[j][1])):min(int(bboxes[j][3])+int(0.05*bboxes[j][3]),1080),max(int(bboxes[j][0])-int(0.05*bboxes[j][0]),0):min(int(bboxes[j][2])+int(0.05*bboxes[j][2]),1920),:]
        crop = img_temp[int(bboxes[j][1]):int(bboxes[j][3]),int(bboxes[j][0]):int(bboxes[j][2]),:]
        gname = str(counter) + '.jpg'
        fname = OUTPUT_FOLDER+"/"+gname
        cv2.imwrite(fname,crop)   
        mf.write(str(counter) + " " + str(counter) + " 1 " + fname + "\n")
mf.close()
        
with open(BBOX_FILE,"w") as f:
    json.dump(all_bboxes,f)
f.close()

# # Prints the bboxes on the images along with the labels
# for i in range(len(img_files)):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     img = cv2.imread(img_files[i])
#     folder = OUTPUT_FOLDER
#     with open(json_files[i]) as f:
#         text = f.read()
#         data = json.loads(text)
#         bboxes = data["bboxes"]
#         labels = data["pred_classes"]
#     f.close()
#     for j in range(len(labels)):
#         cat = class_labels[labels[j]]
#         if cat not in ['zebra','horse','cat','cow','dog']:
#             continue
#         cv2.rectangle(img, (int(bboxes[j][0]),int(bboxes[j][1])), (int(bboxes[j][2]),int(bboxes[j][3])), (0,0,255), 10) 
#         cv2.putText(img, cat, (int(bboxes[j][0]), int(bboxes[j][3])), font, 2, (0,0,0), 2, cv2.LINE_AA)
#     fname = folder+"/"+img_files[i].split('/')[-1].split('.')[0]+".jpg"
#     cv2.imwrite(fname,img)
