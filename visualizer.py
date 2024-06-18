# Output: (Example: data[0])
# ├── sbj_boxes (array): all detected subjects and their locations, shape (num_sbj, boxes_location), example: (600, 4)
# ├── sbj_labels (array): all label of detected subjects, shape (label_sbj,), example: (600,) 
# ├── sbj_scores (array): confidence score of all detected subjects, shape (score_sbj,), example (600, )
# ├── obj_boxes (array): all detected objects and their locations, shape (num_obj, boxes_location), example: (600, 4)
# ├── obj_labels (array): all label of detected objects, shape (label_obj,), example: (600,) 
# ├── obj_scores (array): confidence score of all detected objects, shape (score_obj,), example (600, )
# ├── prd_scores (array): probability distribution of all predicates , shape (prd_scores, num_label_prd), example: (600, 71) 
# ├── image (str): path to the directory of image, example: 'data/vrd/val_images/000000000001.jpg'
# ├── gt_sbj_boxes (array): ground truth subject boxes, shape (num_sbj, boxes_location), example: (14, 4)
# ├── gt_sbj_labels (array): grouth truth subject labels, shape (label_sbj,), example: (14, )
# ├── gt_obj_boxes (array): grouth truth object boxes, shape (num_obj, boxes_location), example: (14, 4)
# ├── gt_obj_labels (array): grouth truth object labels, shape (label_obj,), example: (14,) 
# ├── gt_prd_labels (array): grouth truth predicate labels, shape (predicates_label,), example: (14, )

import pickle
import cv2
import json
import numpy as np

rel_detections_ckpt_path = "Outputs/e2e_relcnn_VGG16_8_epochs_vrd_y_loss_only/rel_detections.pkl"
output_image_path = "test/output_image.jpg"
object_labels_path = "vrd/data/vrd/objects.json"
predicate_labels_path = "vrd/data/vrd/predicates.json"

with open(rel_detections_ckpt_path, "rb") as f: 
    data = pickle.load(f) 

white_color = (255, 255, 255, 255)

def draw_boxes(image, sbj_boxes, sbj_labels, obj_boxes, obj_labels, prd_labels):
    with open(object_labels_path, "r") as f:
        object_labels = json.load(f)
    with open(predicate_labels_path, "r") as f:
        predicates = json.load(f)

    overlay = image.copy()  # Create a copy of the original image for overlay
    drawn_prds = set()
    drawn_sbjs = set()
    drawn_objs = set()

    for i, (sbj_box, sbj_label, obj_box, obj_label, prd_label) in enumerate(zip(sbj_boxes, sbj_labels, obj_boxes, obj_labels, prd_labels)):
        x1_sbj, y1_sbj, x2_sbj, y2_sbj = map(int, sbj_box)
        sbj_center_x = (x1_sbj + x2_sbj) // 2
        sbj_center_y = (y1_sbj + y2_sbj) // 2

        x1_obj, y1_obj, x2_obj, y2_obj = map(int, obj_box)
        obj_center_x = (x1_obj + x2_obj) // 2
        obj_center_y = (y1_obj + y2_obj) // 2

        pred_center_x = (sbj_center_x + obj_center_x) // 2
        pred_center_y = (sbj_center_y + obj_center_y) // 2

        color = (np.random.uniform(0, 255), np.random.uniform(0, 255), np.random.uniform(0, 255))

        # Draw lines on overlay
        cv2.line(overlay, (sbj_center_x, sbj_center_y), (obj_center_x, obj_center_y), color, 2)

        pred_text_x = pred_center_x - 50
        pred_text_y = pred_center_y - 10

        print(f"Subject: {object_labels[sbj_label]}, Object: {object_labels[obj_label]}, Predicate: {predicates[prd_label]}")

        prd_tuple = (tuple(sbj_box), prd_label, tuple(obj_box))
        if prd_tuple not in drawn_prds:
            drawn_prds.add(prd_tuple)
            predicate_label = predicates[prd_label]
            pred_label_size, _ = cv2.getTextSize(predicate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(overlay, (pred_text_x, pred_text_y - pred_label_size[1]), (pred_text_x + pred_label_size[0], pred_text_y), color, -1)
            cv2.putText(overlay, predicate_label, (pred_text_x, pred_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        sbj_tuple = (tuple(sbj_box), sbj_label)
        if sbj_tuple not in drawn_sbjs:
            drawn_sbjs.add(sbj_tuple)
            sbj_label_text = object_labels[sbj_label]
            sbj_text_x = sbj_center_x - 50
            sbj_text_y = sbj_center_y - 10
            sbj_label_size, _ = cv2.getTextSize(sbj_label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(overlay, (sbj_text_x, sbj_text_y - sbj_label_size[1]), (sbj_text_x + sbj_label_size[0], sbj_text_y), color, -1)
            cv2.putText(overlay, sbj_label_text, (sbj_text_x, sbj_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        obj_tuple = (tuple(obj_box), obj_label)
        if obj_tuple not in drawn_objs:
            drawn_objs.add(obj_tuple)
            obj_label_text = object_labels[obj_label]
            obj_text_x = obj_center_x - 50
            obj_text_y = obj_center_y - 10
            obj_label_size, _ = cv2.getTextSize(obj_label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(overlay, (obj_text_x, obj_text_y - obj_label_size[1]), (obj_text_x + obj_label_size[0], obj_text_y), color, -1)
            cv2.putText(overlay, obj_label_text, (obj_text_x, obj_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Blend overlay onto original image
    opacity = 0.6  # Adjust opacity as needed
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

data = data[60]

gt_sbj_boxes = data['gt_sbj_boxes']
gt_sbj_labels = data['gt_sbj_labels']
gt_obj_boxes = data['gt_obj_boxes']
gt_obj_labels = data['gt_obj_labels']
gt_prd_labels = data['gt_prd_labels']

image_path = data['image'].replace('/workspace/Large-Scale-VRD.pytorch/data', 'vrd/data')

image = cv2.imread(image_path)

draw_boxes(image, gt_sbj_boxes, gt_sbj_labels, gt_obj_boxes, gt_obj_labels, gt_prd_labels)

cv2.imwrite(output_image_path, image)

print(f"Image saved to {output_image_path}")