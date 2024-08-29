from itertools import groupby

import torch
import cv2
import numpy as np

from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.augmentations import letterbox
from utils.segment.general import process_mask, scale_masks
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from instill.helpers.const import DataType
from instill.helpers.ray_io import serialize_byte_tensor, deserialize_bytes_tensor
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    construct_infer_response,
    construct_metadata_response,
    Metadata,
)


@instill_deployment
class StomataYolov7:
    def __init__(self):
        self.label = ["stomata", "outer_line"]
        self.device = select_device("cpu")
        self.model = DetectMultiBackend(
            "model-multilabel.pt", device=self.device, dnn=False, data=None, fp16=False
        )

        self.image_size = check_img_size(640, s=self.model.stride)

        self.model.warmup()  # warmup

    def ModelMetadata(self, req):
        resp = construct_metadata_response(
            req=req,
            inputs=[
                Metadata(
                    name="input",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                Metadata(
                    name="rles",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1],
                ),
                Metadata(
                    name="boxes",
                    datatype=str(DataType.TYPE_FP32),
                    shape=[-1, 4],
                ),
                Metadata(
                    name="labels",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1],
                ),
                Metadata(
                    name="scores",
                    datatype=str(DataType.TYPE_FP32),
                    shape=[-1],
                ),
            ],
        )
        return resp

    def rle_encode(self, binary_mask):
        r"""
        Args:
            binary_mask: a binary mask with the shape of `mask_shape`

        Returns uncompressed Run-length Encoding (RLE) in COCO format
                Link: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
                {
                    'counts': [n1, n2, n3, ...],
                    'size': [height, width] of the mask
                }
        """
        fortran_binary_mask = np.asfortranarray(binary_mask)
        uncompressed_rle = {"counts": [], "size": list(binary_mask.shape)}
        counts = uncompressed_rle.get("counts")
        for i, (value, elements) in enumerate(
            groupby(fortran_binary_mask.ravel(order="F"))
        ):
            if i == 0 and value == 1:
                counts.append(
                    0
                )  # Add 0 if the mask starts with one, since the odd counts are always the number of zeros
            counts.append(len(list(elements)))

        return uncompressed_rle

    def post_process(self, boxes, labels, masks, scores, score_threshold=0.7):
        rles = []
        ret_boxes = []
        ret_scores = []
        ret_labels = []
        for mask, box, label, score in zip(masks, boxes, labels, scores):
            box = box.cpu()
            score = score.cpu()
            # Showing boxes with score > 0.7
            if score <= score_threshold:
                continue
            ret_scores.append(score)
            ret_labels.append(label)
            int_box = [int(i) for i in box]
            mask = mask[int_box[1] : int_box[3] + 1, int_box[0] : int_box[2] + 1]
            ret_boxes.append(
                [
                    int_box[0],
                    int_box[1],
                    int_box[2] - int_box[0] + 1,
                    int_box[3] - int_box[1] + 1,
                ]
            )  # convert to x,y,w,h
            mask = mask > 0.5
            rle = self.rle_encode(mask).get("counts")
            rle = [str(i) for i in rle]
            rle = ",".join(
                rle
            )  # output batching need to be same shape then convert rle to string for each object mask
            rles.append(rle)

        return rles, ret_boxes, ret_labels, ret_scores

    async def __call__(self, req):
        b_tensors = req.raw_input_contents[0]

        input_tensors = deserialize_bytes_tensor(b_tensors)

        image_masks = []
        image_boxes = []
        image_scores = []
        image_labels = []
        for enc in input_tensors:
            frame = cv2.imdecode(np.frombuffer(enc, np.uint8), cv2.IMREAD_COLOR)
            im = letterbox(frame, self.image_size, stride=self.model.stride)[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred, out = self.model(im)
            proto = out[1]
            pred = non_max_suppression(
                pred,
                conf_thres=0.3,
                max_det=1000,
                nm=32,
            )

            det_masks = []
            labels = []
            for i, det in enumerate(pred):  # per image
                if len(det):  # per detection
                    masks = process_mask(
                        proto[i], det[:, 6:], det[:, :4], im.shape[2:], True
                    )  # HWC
                    masks = scale_masks(
                        im.shape[2:],
                        masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                        frame.shape,
                    )
                    masks = np.transpose(masks, (2, 0, 1))
                    det[:, :4] = scale_coords(
                        im.shape[2:], det[:, :4], frame.shape
                    ).round()

                    det_masks.extend(masks)
                    labels.extend([self.label[int(d)] for d in det[:, 5]])

            image_masks.append(det_masks)
            image_boxes.append(pred[0][:, :4])
            image_scores.append(pred[0][:, 4])
            image_labels.append(labels)

        # og post process
        rs_boxes = []
        rs_labels = []
        rs_rles = []
        rs_scores = []

        for boxes, labels, masks, scores in zip(
            image_boxes, image_labels, image_masks, image_scores
        ):  # single image
            o_rles, o_boxes, o_labels, o_scores = self.post_process(
                boxes, labels, masks, scores
            )
            rs_boxes.append(o_boxes)
            rs_labels.append(o_labels)
            rs_scores.append(o_scores)
            rs_rles.append(o_rles)

        max_boxes = max([len(i) for i in rs_boxes])
        for b in rs_boxes:
            for _ in range(max_boxes - len(b)):
                b.append([0, 0, 0, 0])
        max_labels = max([len(i) for i in rs_labels])
        for lb in rs_labels:
            for _ in range(max_labels - len(lb)):
                lb.append("")
        max_scores = max([len(i) for i in rs_scores])
        for sc in rs_scores:
            for _ in range(max_scores - len(sc)):
                sc.append(0)

        max_rles = max([len(i) for i in rs_rles])
        for rles in rs_rles:
            for _ in range(max_rles - len(rles)):
                rles.append("")

        resp_outputs = []
        resp_raw_outputs = []

        # rles
        resp_outputs.append(
            Metadata(
                name="rles",
                shape=[len(input_tensors), len(rs_rles[0])],
                datatype=str(DataType.TYPE_STRING),
            )
        )
        rles_out = []
        for r in rs_rles:
            rles_out.extend(r)
        if len(rles_out) != 0:
            rles_out = [bytes(f"{rles_out[i]}", "utf-8") for i in range(len(rles_out))]
            resp_raw_outputs.append(serialize_byte_tensor(np.asarray(rles_out)))
        else:
            resp_raw_outputs.append(b"")

        # boxes
        resp_outputs.append(
            Metadata(
                name="boxes",
                shape=[len(input_tensors), len(rs_boxes[0]), 4],
                datatype=str(DataType.TYPE_FP32),
            )
        )
        resp_raw_outputs.append(np.asarray(rs_boxes).astype(np.float32).tobytes())

        # labels
        resp_outputs.append(
            Metadata(
                name="labels",
                shape=[len(input_tensors), len(rs_labels[0])],
                datatype=str(DataType.TYPE_STRING),
            )
        )
        labels_out = []
        for r in rs_labels:
            labels_out.extend(r)
        if len(labels_out) != 0:
            labels_out = [
                bytes(f"{labels_out[i]}", "utf-8") for i in range(len(labels_out))
            ]
            resp_raw_outputs.append(serialize_byte_tensor(np.asarray(labels_out)))
        else:
            resp_raw_outputs.append(b"")

        # scores
        resp_outputs.append(
            Metadata(
                name="scores",
                shape=[len(input_tensors), len(rs_scores[0])],
                datatype=str(DataType.TYPE_FP32),
            )
        )
        resp_raw_outputs.append(np.asarray(rs_scores).astype(np.float32).tobytes())

        return construct_infer_response(
            req=req,
            outputs=resp_outputs,
            raw_outputs=resp_raw_outputs,
        )


entrypoint = InstillDeployable(StomataYolov7).get_deployment_handle()
