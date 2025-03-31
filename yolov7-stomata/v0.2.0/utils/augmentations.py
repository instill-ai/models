import cv2
import torch
import torch.nn.functional as F
import numpy as np


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def letterbox_tensor(
    im: torch.Tensor,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
) -> tuple:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[1:3]  # current shape [channels, height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = (
        int(round(shape[0] * r)),
        int(round(shape[1] * r)),
    )  # new unpadded size
    dh, dw = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = dw % stride, dh % stride  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        # Resize using interpolation (nearest or linear)
        im = F.interpolate(
            im.unsqueeze(0), size=new_unpad, mode="bilinear", align_corners=False
        ).squeeze(0)

    # Calculate padding values
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = F.pad(
        im, (top, bottom, left, right), mode="constant", value=color[0] / 255
    )  # add border

    return im, ratio, (dw, dh)


def expand_and_shift_masks(masks, target_shape=(640, 640), x_offset=0, y_offset=0):
    """
    Expand and shift segmentation masks.

    Args:
        masks (torch.Tensor): Tensor of shape (N, H, W), where N is the number of masks,
                              H and W are the height and width of each mask.
        target_shape (tuple): Desired shape to expand to, in the form (height, width).
        x_offset (int): Horizontal offset to shift the mask area.
        y_offset (int): Vertical offset to shift the mask area.

    Returns:
        torch.Tensor: Modified tensor of masks with the expanded and shifted segment areas.
    """
    # Apply padding to shift mask
    masks = F.pad(masks, (x_offset, 0, y_offset, 0), mode="constant", value=0)

    # Current mask height and width
    _, original_height, original_width = masks.shape
    target_height, target_width = target_shape

    # Padding along the bottom and right
    pad_bottom = max(0, target_height - original_height)
    pad_right = max(0, target_width - original_width)

    # Apply padding to target shape
    masks = F.pad(masks, (0, pad_right, 0, pad_bottom), mode="constant", value=0)

    return masks
