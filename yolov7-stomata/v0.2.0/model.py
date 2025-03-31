import torch
import numpy as np

from itertools import groupby
from utils.general import clip_coords, non_max_suppression, scale_coords, check_img_size
from utils.augmentations import letterbox, letterbox_tensor, expand_and_shift_masks
from utils.segment.general import process_mask, scale_masks
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    parse_task_instance_segmentation_to_vision_input,
    construct_task_instance_segmentation_output,
)


@instill_deployment
class StomataYolov7:
    def __init__(self):
        self.label = ["pore", "stomata"]
        self.device = select_device("cuda:0")
        self.model_stage1 = DetectMultiBackend(
            "outerlinebest.pt", device=self.device, dnn=False, data=None
        )
        self.model_stage2 = DetectMultiBackend(
            "porebest.pt", device=self.device, dnn=False, data=None
        )

        self.image_size_stage1 = check_img_size(640, s=self.model_stage1.stride)
        self.image_size_stage2 = check_img_size(640, s=self.model_stage2.stride)

        self.model_stage1.warmup()  # warmup
        self.model_stage2.warmup()  # warmup

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
            # Showing boxes with score > 0.7
            if label == self.label[1] and score <= score_threshold:
                continue
            if label == self.label[0] and score <= 0.3:
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

    async def __call__(self, request):
        vision_inputs = await parse_task_instance_segmentation_to_vision_input(
            request=request
        )

        image_masks = []
        image_boxes = []
        image_scores = []
        image_labels = []
        for inp in vision_inputs:
            # frame = cv2.imdecode(np.frombuffer(enc, np.uint8), cv2.IMREAD_COLOR)
            frame = np.array(inp.image.convert("RGB"), dtype=np.float32)
            im = letterbox(
                frame,
                self.image_size_stage1,
                stride=self.model_stage1.stride,
            )[0]
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.model_stage1.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred, out = self.model_stage1(im)
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

                det_c = det.clone()
                clip_coords(det_c[:, :4], im.shape[2:])

                pred_stage2_list = []
                pred_mask_stage2_list = []
                for ii, box in enumerate(det_c[:, :4]):
                    if det_c[ii, 4] < 0.7:
                        continue
                    x1, y1, x2, y2 = torch.round(box).to(torch.int)
                    cropped_image_original = im[i][:, y1:y2, x1:x2]
                    cropped_image, ratio_stage2, pad_stage2 = letterbox_tensor(
                        im=cropped_image_original,
                        new_shape=self.image_size_stage2,
                        stride=self.model_stage2.stride,
                    )
                    cropped_image = cropped_image[None]
                    p_stage2, train_out_stage2 = self.model_stage2(cropped_image)
                    p_stage2 = non_max_suppression(
                        p_stage2,
                        conf_thres=0.3,
                        max_det=4,
                        nm=32,
                    )

                    for si, pred_stage2 in enumerate(p_stage2):
                        proto_out_stage2 = train_out_stage2[1][si]
                        if pred_stage2.shape[0] == 0:
                            continue

                        pred_stage2n = pred_stage2.clone()
                        # Predictions
                        ## to stage 1 inference space pred
                        scale_coords(
                            cropped_image[si].shape[1:],
                            pred_stage2[:, :4],
                            cropped_image_original.shape[1:],
                        )
                        ## transform pred
                        offsets = torch.tensor([x1, y1, x1, y1]).to(
                            self.device, non_blocking=True
                        )
                        pred_stage2[:, :4] += offsets

                        ## to stage 1 native-space pred
                        scale_coords(
                            im.shape[2:],
                            pred_stage2[:, :4],
                            frame.shape,
                        ).round()

                        # Masks
                        pred_masks_stage2 = process_mask(
                            proto_out_stage2,
                            pred_stage2n[:, 6:],
                            pred_stage2n[:, :4],
                            shape=cropped_image[si].shape[1:],
                            upsample=True,
                        )

                        pred_masks_stage2 = scale_masks(
                            cropped_image[si].shape[1:],
                            pred_masks_stage2.permute(1, 2, 0),
                            [
                                cropped_image_original.shape[1],
                                cropped_image_original.shape[2],
                                3,
                            ],
                        )

                        # transform mask
                        pred_masks_stage2 = expand_and_shift_masks(
                            pred_masks_stage2,
                            x_offset=x1,
                            y_offset=y1,
                            target_shape=im.shape[2:],
                        )

                        pred_masks_stage2 = scale_masks(
                            im.shape[2:],
                            pred_masks_stage2.permute(1, 2, 0),
                            frame.shape,
                        )

                        pred_stage2_list.append(pred_stage2.cpu().numpy())
                        pred_mask_stage2_list.append(pred_masks_stage2.cpu().numpy())

                if len(det):
                    out_stage2 = np.concatenate(pred_stage2_list, axis=0)
                    out_mask_stage2 = np.concatenate(pred_mask_stage2_list, axis=0)

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
                        im.shape[2:],
                        det[:, :4],
                        frame.shape,
                    ).round()
                    det[:, 5] = 1

                    final_pred = np.concatenate((det.cpu().numpy(), out_stage2), axis=0)
                    final_pred_masks = np.concatenate((masks, out_mask_stage2), axis=0)

                    det_masks.extend(final_pred_masks)

                    labels.extend([self.label[int(d)] for d in final_pred[:, 5]])

            image_masks.append(det_masks)
            image_boxes.append(final_pred[:, :4])
            image_scores.append(final_pred[:, 4])
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

        return construct_task_instance_segmentation_output(
            request, rs_rles, rs_labels, rs_scores, rs_boxes
        )


entrypoint = InstillDeployable(StomataYolov7).get_deployment_handle()
