import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from darknet import Darknet
import torch
from torch.autograd import Variable

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor']


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/whole_model_trained_yolo_voc")
    parser.add_argument("--input", type=str, default="test_images")
    parser.add_argument("--output", type=str, default="test_images")

    args = parser.parse_args()
    return args


def post_processing(logits, image_size, gt_classes, anchors, conf_threshold, nms_threshold):
    num_anchors = len(anchors)
    anchors = torch.Tensor(anchors)
    if isinstance(logits, torch.Variable):
        logits = logits.data

    if logits.dim() == 3:
        logits.unsqueeze_(0)

    batch = logits.size(0)
    h = logits.size(2)
    w = logits.size(3)

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
    lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)
    if torch.cuda.is_available():
        lin_x = lin_x.cuda()
        lin_y = lin_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

    logits = logits.view(batch, num_anchors, -1, h * w)
    logits[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)
    logits[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)
    logits[:, :, 2, :].exp_().mul_(anchor_w).div_(w)
    logits[:, :, 3, :].exp_().mul_(anchor_h).div_(h)
    logits[:, :, 4, :].sigmoid_()

    with torch.no_grad():
        cls_scores = torch.nn.functional.softmax(logits[:, :, 5:, :], 2)
    cls_max, cls_max_idx = torch.max(cls_scores, 2)
    cls_max_idx = cls_max_idx.float()
    cls_max.mul_(logits[:, :, 4, :])

    score_thresh = cls_max > conf_threshold
    score_thresh_flat = score_thresh.view(-1)

    if score_thresh.sum() == 0:
        predicted_boxes = []
        for i in range(batch):
            predicted_boxes.append(torch.Tensor([]))
    else:
        coords = logits.transpose(2, 3)[..., 0:4]
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = cls_max[score_thresh]
        idx = cls_max_idx[score_thresh]
        detections = torch.cat([coords, scores[:, None], idx[:, None]], dim=1)

        max_det_per_batch = num_anchors * h * w
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(batch)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        predicted_boxes = []
        start = 0
        for end in split_idx:
            predicted_boxes.append(detections[start: end])
            start = end

    selected_boxes = []
    for boxes in predicted_boxes:
        if boxes.numel() == 0:
            return boxes

        a = boxes[:, :2]
        b = boxes[:, 2:4]
        bboxes = torch.cat([a - b / 2, a + b / 2], 1)
        scores = boxes[:, 4]

        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > nms_threshold).triu(1)

        keep = conflicting.sum(0).byte()
        keep = keep.cpu()
        conflicting = conflicting.cpu()

        keep_len = len(keep) - 1
        for i in range(1, keep_len):
            if keep[i] > 0:
                keep -= conflicting[i]
        if torch.cuda.is_available():
            keep = keep.cuda()

        keep = (keep == 0)
        selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 6).contiguous())

    final_boxes = []
    for boxes in selected_boxes:
        if boxes.dim() == 0:
            final_boxes.append([])
        else:
            boxes[:, 0:3:2] *= image_size
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1:4:2] *= image_size
            boxes[:, 1] -= boxes[:, 3] / 2

            final_boxes.append([[box[0].item(), box[1].item(), box[2].item(), box[3].item(), box[4].item(),
                                 gt_classes[int(box[5].item())]] for box in boxes])
    return final_boxes

def test(opt):
    model = Darknet('./cfg/yolo.cfg')
    model.load_weights('data/yolov2/yolo.weights')
    model = model.eval()
    model = model.cuda()

    colors = pickle.load(open("./pallete", "rb"))

    for image_path in glob.iglob(opt.input + os.sep + '*.jpg'):
        if "prediction" in image_path:
            continue
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        image = image[None, :, :, :]
        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height
        data = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            logits = model(data)
            predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                          opt.nms_threshold)
        if len(predictions) != 0:
            predictions = predictions[0]
            output_image = cv2.imread(image_path)
            for pred in predictions:
                xmin = int(max(pred[0] / width_ratio, 0))
                ymin = int(max(pred[1] / height_ratio, 0))
                xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
                ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
                color = colors[CLASSES.index(pred[5])]
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, pred[5] + ' : %.2f' % pred[4],
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
                print("Object: {}, Bounding box: ({},{}) ({},{})".format(pred[5], xmin, xmax, ymin, ymax))
            cv2.imwrite(image_path[:-4] + "_prediction.jpg", output_image)


if __name__ == "__main__":
    opt = get_args()
    test(opt)