import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from .ip_adapter.instantId import CrossAttentionPatch


def draw_kps(w, h, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    """
    Draws keypoints and limb connections on an image canvas.

    Args:
        w (int): Width of the output image.
        h (int): Height of the output image.
        kps (list): List of keypoints, where each keypoint is (x, y).
        color_list (list, optional): List of colors for keypoints and limbs. Defaults to preset colors.

    Returns:
        np.ndarray: The output image with drawn keypoints and limb connections.
    """
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)
    out_img = np.zeros([h, w, 3])

    # Draw limb connections
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]
        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) **  2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    # Draw keypoints
    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    return out_img.astype(np.uint8)


def set_model_patch_replace(model, patch_kwargs, key):
    """
    Sets or updates the patch replacement condition in a model's transformer options.

    Args:
        model (object): The model instance whose transformer options will be modified.
        patch_kwargs (dict): Keyword arguments to define or update the patch replacement.
        key (tuple): A key identifying the specific patch to replace.

    Returns:
        None
    """
    attn = "attn2"
    to = model.model_options["transformer_options"].copy()
    
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()
    
    if attn not in to["patches_replace"]:
        to["patches_replace"][attn] = {}
    else:
        to["patches_replace"][attn] = to["patches_replace"][attn].copy()
    
    if key not in to["patches_replace"][attn]:
        to["patches_replace"][attn][key] = CrossAttentionPatch(**patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"][attn][key].set_new_condition(**patch_kwargs)


def resize_to_fit_area(original_width, original_height, area_width, area_height):
    """
    Resizes an image while maintaining aspect ratio to fit within a specified area.

    Args:
        original_width (int): Original width of the image.
        original_height (int): Original height of the image.
        area_width (int): Maximum allowed width.
        area_height (int): Maximum allowed height.

    Returns:
        tuple: New width and height that fit within the specified area.
    """
    base_pixels = 8
    max_area = area_width * area_height
    aspect_ratio = original_width / original_height

    scale_factor = math.sqrt(max_area / (original_width * original_height))
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    new_width = new_width // base_pixels * base_pixels
    new_height = new_height // base_pixels * base_pixels

    if new_width * new_height > max_area:
        new_width = math.floor(math.sqrt(max_area * aspect_ratio)) // base_pixels * base_pixels
        new_height = math.floor(new_width / aspect_ratio) // base_pixels * base_pixels

    return new_width, new_height


def get_mask_bbox_with_padding(mask_image, pad_top, pad_right, pad_bottom, pad_left):
    """
    Extracts the bounding box of the masked region, applying additional padding.

    Args:
        mask_image (torch.Tensor): Binary mask where 1 represents the object.
        pad_top (int): Padding to add to the top of the mask.
        pad_right (int): Padding to add to the right.
        pad_bottom (int): Padding to add to the bottom.
        pad_left (int): Padding to add to the left.

    Returns:
        tuple: Padded bounding box coordinates (x1, y1, x2, y2).
    """
    mask_segments = torch.nonzero(mask_image == 1, as_tuple=False)
    if torch.count_nonzero(mask_segments).item() == 0:
        raise Exception("Draw a mask on pose image")

    m_y1, m_y2 = torch.min(mask_segments[:, 0]).item(), torch.max(mask_segments[:, 0]).item()
    m_x1, m_x2 = torch.min(mask_segments[:, 1]).item(), torch.max(mask_segments[:, 1]).item()

    height, width = mask_image.shape

    p_x1 = max(0, m_x1 - pad_left)
    p_y1 = max(0, m_y1 - pad_top)
    p_x2 = min(width, m_x2 + pad_right)
    p_y2 = min(height, m_y2 + pad_bottom)
    
    return int(p_x1), int(p_y1), int(p_x2), int(p_y2)


def get_kps_from_image(image, insightface):
    """
    Extracts facial keypoints from an image using InsightFace.

    Args:
        image (torch.Tensor): Input image.
        insightface (INSIGHTFACE_APP): InsightFace model instance.

    Returns:
        list: List of facial keypoints.
    """
    np_pose_image = (255.0 * image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
    face_info = insightface.get(cv2.cvtColor(np_pose_image, cv2.COLOR_RGB2BGR))
    assert len(face_info) > 0, "No face detected in pose image"
    face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
    return face_info["kps"]


def get_angle(a=(0, 0), b=(0, 0), round_angle=False):
    """
    Computes the angle between two points, typically the eyes.

    Args:
        a (tuple): Coordinates of the first point (default: left eye).
        b (tuple): Coordinates of the second point (default: right eye).
        round_angle (bool): If True, rounds the angle to the nearest 90 degrees.

    Returns:
        float: Computed rotation angle in degrees.
    """
    angle = math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / math.pi
    if round_angle:
        angle = round(angle / 90) * 90
        if angle == 360:
            angle = 0
    return angle


def rotate_with_pad(image, clockwise, angle):
    """
    Rotates an image with padding to prevent cropping.

    Args:
        image (torch.Tensor): Input image.
        clockwise (bool): Whether to rotate clockwise or counterclockwise.
        angle (float): Rotation angle in degrees.

    Returns:
        torch.Tensor: Rotated image with padding.
    """
    if not clockwise:
        angle *= -1
        
    image = image.squeeze(0)
    image = image.permute(2, 0, 1)
    image = TF.rotate(image, angle, fill=0, expand=True)
    image = image.permute(1, 2, 0)
    image = image.unsqueeze(0)
    return image