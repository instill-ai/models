import torch
import time
import requests
import cv2
import numpy as np
import onnxruntime as ort
import torchvision
from typing import List
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_detection_to_vision_input,
    construct_task_detection_output,
)


@instill_deployment
class Yolov7:
    def __init__(self):
        self.categories = self._image_labels()
        self.model = ort.InferenceSession("model.onnx")

    def _image_labels(self) -> List[str]:
        categories = []
        url = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt"
        labels = requests.get(url, timeout=10).text
        for label in labels.split("\n"):
            categories.append(label.strip())
        return categories

    def _pre_procoess(self, vision_inputs):
        def get_preprocess_size(image, target_size, stride=32):
            ih, iw = target_size
            h, w, _ = np.array(image).shape

            scale = min(iw / w, ih / h)
            nw, nh = int(round(scale * w)), int(round(scale * h))
            dw, dh = iw - nw, ih - nh  # wh padding
            # wh padding to the closest value dividable by stride
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
            # scaled with padding that is dividable by stride
            scaled_w, scaled_h = nw + dw, nh + dh

            return scaled_h, scaled_w, h, w

        def image_preprocess_for_batch(image, batch_h, batch_w):
            h, w, _ = np.array(image).shape

            scale = min(batch_w / w, batch_h / h)
            nw, nh = int(round(scale * w)), int(round(scale * h))
            dw, dh = batch_w - nw, batch_h - nh  # wh padding
            dw /= 2  # divide padding into 2 sides
            dh /= 2

            top = int(round(dh - 0.1))
            left = int(round(dw - 0.1))

            image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            image_resized = np.array(image_resized)  # h x w x c
            image_resized = np.transpose(
                image_resized, axes=(2, 0, 1)
            )  # convert to c x h x w

            image_padded = np.full(shape=[3, batch_h, batch_w], fill_value=114.0)
            image_padded[:, top : nh + top, left : nw + left] = image_resized
            image_padded = np.ascontiguousarray(image_padded)

            image_padded = image_padded / 255.0

            return image_padded

        images = []
        orig_img_hw = []
        scaled_img_hw = []

        batch_images = []
        batch_w, batch_h = -1, -1
        for vision_input in vision_inputs:
            pil_img = vision_input.image
            image = np.array(pil_img)
            if len(image.shape) == 2:  # gray image
                raise ValueError(
                    f"The image shape with {image.shape} is " f"not in acceptable"
                )
            scaled_h, scaled_w, orig_h, orig_w = get_preprocess_size(image, [640, 640])
            batch_w = max(batch_w, scaled_w)
            batch_h = max(batch_h, scaled_h)
            batch_images.append(image)
            orig_img_hw.append([orig_h, orig_w])

        for img in batch_images:
            image_data = image_preprocess_for_batch(img, batch_h, batch_w)
            image_data = image_data.astype(float)
            images.append(image_data)
            scaled_img_hw.append([batch_h, batch_w])

        return images, orig_img_hw, scaled_img_hw

    def _post_process(self, outputs, orig_img_hw, scaled_img_hw):
        def box_iou(box1, box2):
            # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
            """
            Return intersection-over-union (Jaccard index) of boxes.
            Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
            Arguments:
                box1 (Tensor[N, 4])
                box2 (Tensor[M, 4])
            Returns:
                iou (Tensor[N, M]): the NxM matrix containing the pairwise
                    IoU values for every element in boxes1 and boxes2
            """

            def box_area(box):
                # box = 4xn
                return (box[2] - box[0]) * (box[3] - box[1])

            area1 = box_area(box1.T)
            area2 = box_area(box2.T)

            # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
            inter = (
                (
                    torch.min(box1[:, None, 2:], box2[:, 2:])
                    - torch.max(box1[:, None, :2], box2[:, :2])
                )
                .clamp(0)
                .prod(2)
            )
            # iou = inter / (area1 + area2 - inter)
            return inter / (area1[:, None] + area2 - inter)

        def scale_coords(img1_hw, coords, img0_hw, ratio_pad=None):
            # Rescale coords (xyxy) from img1_hw to img0_hw shape
            if ratio_pad is None:  # calculate from img0_shape
                gain = min(
                    img1_hw[0] / img0_hw[0], img1_hw[1] / img0_hw[1]
                )  # gain  = old / new
                pad = (img1_hw[1] - img0_hw[1] * gain) / 2, (
                    img1_hw[0] - img0_hw[0] * gain
                ) / 2  # wh padding
            else:
                gain = ratio_pad[0][0]
                pad = ratio_pad[1]

            coords[:, [0, 2]] -= pad[0]  # x padding
            coords[:, [1, 3]] -= pad[1]  # y padding
            coords[:, :4] /= gain
            clip_coords(coords, img0_hw)
            return coords

        def clip_coords(boxes, img_shape):
            # Clip bounding xyxy bounding boxes to image shape (height, width)
            boxes[:, 0].clamp_(0, img_shape[1])  # x1
            boxes[:, 1].clamp_(0, img_shape[0])  # y1
            boxes[:, 2].clamp_(0, img_shape[1])  # x2
            boxes[:, 3].clamp_(0, img_shape[0])  # y2

        def xywh2xyxy(x):
            # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
            y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
            y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
            y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
            return y

        def xyxy2xywh(x):
            # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where x=left, x=top
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[:, 0] = x[:, 0]  # top left x
            y[:, 1] = x[:, 1]  # top left y
            y[:, 2] = x[:, 2] - x[:, 0]  # width
            y[:, 3] = x[:, 3] - x[:, 1]  # height
            return y

        def non_max_suppression(
            prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
        ):
            """Runs Non-Maximum Suppression (NMS) on inference results

            Returns:
                list of detections, on (n,6) tensor per image [xyxy, conf, cls]
            """

            nc = prediction.shape[2] - 5  # number of classes
            xc = prediction[..., 4] > conf_thres  # candidates

            # Settings
            # (pixels) minimum and maximum box width and height
            min_wh, max_wh = 2, 4096
            max_det = 300  # maximum number of detections per image
            max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
            time_limit = 10.0  # seconds to quit after
            redundant = True  # require redundant detections
            multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
            merge = False  # use merge-NMS

            t = time.time()
            output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[
                0
            ]
            for xi, x in enumerate(prediction):  # image index, image inference
                # Apply constraints
                # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
                x = x[xc[xi]]  # confidence

                # Cat apriori labels if autolabelling
                if labels and len(labels[xi]):
                    l = labels[xi]
                    v = torch.zeros((len(l), nc + 5), device=x.device)
                    v[:, :4] = l[:, 1:5]  # box
                    v[:, 4] = 1.0  # conf
                    v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                    x = torch.cat((x, v), 0)

                # If none remain process next image
                if not x.shape[0]:
                    continue

                # Compute conf
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                box = xywh2xyxy(x[:, :4])

                # Detections matrix nx6 (xyxy, conf, cls)
                if multi_label:
                    i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                    x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
                else:  # best class only
                    conf, j = x[:, 5:].max(1, keepdim=True)
                    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

                # Filter by class
                if classes is not None:
                    x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

                # Apply finite constraint
                # if not torch.isfinite(x).all():
                #     x = x[torch.isfinite(x).all(1)]

                # Check shape
                n = x.shape[0]  # number of boxes
                if not n:  # no boxes
                    continue
                elif n > max_nms:  # excess boxes
                    # sort by confidence
                    x = x[x[:, 4].argsort(descending=True)[:max_nms]]

                # Batched NMS
                c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
                # boxes (offset by class), scores
                boxes, scores = x[:, :4] + c, x[:, 4]
                i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
                if i.shape[0] > max_det:  # limit detections
                    i = i[:max_det]
                if merge and (
                    1 < n < 3e3
                ):  # Merge NMS (boxes merged using weighted mean)
                    # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                    iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                    weights = iou * scores[None]  # box weights
                    x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                        1, keepdim=True
                    )  # merged boxes
                    if redundant:
                        i = i[iou.sum(1) > 1]  # require redundancy

                output[xi] = x[i]
                if (time.time() - t) > time_limit:
                    print(f"WARNING: NMS time limit {time_limit}s exceeded")
                    break  # time limit exceeded

            return output

        pred_list = non_max_suppression(torch.from_numpy(np.asarray(outputs)))

        bboxes = []
        labels = []
        scores = []

        for pred, o_hw, s_hw in zip(pred_list, orig_img_hw, scaled_img_hw):
            # Rescale bounding boxes in pred (n, 6) back to original image size
            pred[:, :4] = scale_coords(s_hw, pred[:, :4], o_hw).round()

            # Change from pytorch tensor to numpy array
            pred = pred.numpy()

            if len(pred) > 0:
                bboxes.append(xyxy2xywh(pred[:, :4]))
                scores.append(pred[:, 4])
                labels.append([self.categories[int(idx)] for idx in pred[:, 5]])
            else:
                bboxes.append(np.array([]))
                scores.append([])
                labels.append([])

        return bboxes, scores, labels

    async def __call__(self, request):
        vision_inputs = parse_task_detection_to_vision_input(request=request)

        images, orig_img_hw, scaled_img_hw = self._pre_procoess(vision_inputs)

        images = np.asarray(images, dtype=np.float32)
        outputs = self.model.run(None, {"images": images})

        bboxes, scores, labels = self._post_process(
            outputs[0], orig_img_hw, scaled_img_hw
        )

        return construct_task_detection_output(
            bounding_boxes=bboxes,
            scores=scores,
            categories=labels,
        )


entrypoint = InstillDeployable(Yolov7).get_deployment_handle()
