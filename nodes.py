import os

import comfy.utils
import cv2
import folder_paths
import node_helpers
import numpy as np
import torch
from comfy_execution.graph_utils import GraphBuilder
from insightface.app import FaceAnalysis
from PIL import Image

from .ip_adapter.instantId import InstantId
from .ip_adapter.resampler import Resampler
from .utils import (draw_kps, get_angle, get_kps_from_image,
                    get_mask_bbox_with_padding, resize_to_fit_area,
                    rotate_with_pad, set_model_patch_replace)

folder_paths.folder_names_and_paths["ipadapter"] = ([os.path.join(folder_paths.models_dir, "ipadapter")],
                                                    folder_paths.supported_pt_extensions)
INSIGHTFACE_PATH = os.path.join(folder_paths.models_dir, "insightface")
CATEGORY_NAME = "InstantId Faceswap"
MAX_RESOLUTION = 16384


class FaceEmbed:
    """
    A class responsible for generating face embeddings using InsightFace.
    This allows for facial recognition and identity preservation in deep learning tasks.
    """
    
    def init(self):
        """
        Initializes the FaceEmbed class.
        """
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required and optional input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "insightface": ("INSIGHTFACE_APP",),  # InsightFace model instance
                "face_image": ("IMAGE",)  # Input face image for embedding extraction
            },
            "optional": {
                "face_embeds": ("FACE_EMBED",)  # Optional pre-existing face embeddings
            }
        }

    RETURN_TYPES = ("FACE_EMBED",)  # Output type: face embedding tensor
    RETURN_NAMES = ("face embeds",)  # Output name
    FUNCTION = "make_face_embed"  # Function to be executed
    CATEGORY = CATEGORY_NAME  # UI category

    def make_face_embed(self, insightface, face_image, face_embeds=None):
        """
        Generates a face embedding using InsightFace and optionally concatenates it with existing embeddings.

        Args:
            insightface (INSIGHTFACE_APP): The InsightFace model instance for feature extraction.
            face_image (IMAGE): The input image containing a face.
            face_embeds (Optional[FACE_EMBED]): Previously computed embeddings for concatenation.

        Returns:
            tuple: A tuple containing the generated or updated face embeddings.
        """
        # Convert image tensor to numpy and ensure proper format
        face_image = (255.0 * face_image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
        face_info = insightface.get(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        
        assert len(face_info) > 0, "No face detected for face embedding"
        
        # Select the largest detected face
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        face_emb = torch.tensor(face_info["embedding"], dtype=torch.float32).unsqueeze(0)
        
        # If no previous embeddings exist, return the new embedding
        if face_embeds is None:
            return face_emb,
        
        # Concatenate the new embedding with existing embeddings
        face_embeds = torch.cat((face_embeds, face_emb), dim=-2)
        return (face_embeds,)


class FaceEmbedCombine:
    """
    A class responsible for combining multiple face embeddings using a resampler.
    This allows for generating a single, averaged facial representation for conditioning.
    """
    
    def init(self):
        """
        Initializes the FaceEmbedCombine class.
        """
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "resampler": ("RESAMPLER",),  # Resampler model used for embedding refinement
                "face_embeds": ("FACE_EMBED",)  # Input tensor containing multiple face embeddings
            },
        }

    RETURN_TYPES = ("FACE_CONDITIONING",)  # Output type: face conditioning embedding
    RETURN_NAMES = ("face conditioning",)  # Output name
    FUNCTION = "combine_face_embed"  # Function to be executed
    CATEGORY = CATEGORY_NAME  # UI category

    def combine_face_embed(self, resampler, face_embeds):
        """
        Combines multiple face embeddings into a single representation using a resampler.

        Args:
            resampler (RESAMPLER): The model responsible for processing face embeddings.
            face_embeds (FACE_EMBED): A tensor containing multiple facial embeddings.

        Returns:
            tuple: A tuple containing the final face conditioning embedding.
        """
        # Compute the mean embedding across all inputs
        embeds = torch.mean(face_embeds, dim=0, dtype=torch.float32).unsqueeze(0)
        
        # Reshape the tensor to match the expected input format for the resampler
        embeds = embeds.reshape([1, -1, 512])
        
        # Generate the conditioning embedding and move it to the appropriate device
        conditionings = resampler(embeds).to(comfy.model_management.get_torch_device())
        return (conditionings,)


class AngleFromFace:
    """
    A class responsible for calculating the face rotation angle based on keypoints.
    Supports different rotation modes to determine whether the angle should be rounded.
    """
    
    rotate_modes = ["none", "loseless", "any"]

    def init(self):
        """
        Initializes the AngleFromFace class.
        """
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "insightface": ("INSIGHTFACE_APP",),  # InsightFace model for keypoint extraction
                "image": ("IMAGE", {"tooltip": "Pose image."}),  # Input image containing the face
                "mask": ("MASK",),  # Mask defining the area of interest
                "rotate_mode": (cls.rotate_modes,),  # Rotation mode selection
                "pad_top": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),  # Padding for face cropping (top)
                "pad_right": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),  # Padding for face cropping (right)
                "pad_bottom": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),  # Padding for face cropping (bottom)
                "pad_left": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),  # Padding for face cropping (left)
            },
        }

    RETURN_TYPES = ("FLOAT",)  # Output type: computed rotation angle
    RETURN_NAMES = ("angle",)  # Output name
    FUNCTION = "get_angle"  # Function to be executed
    CATEGORY = CATEGORY_NAME  # UI category

    def get_angle(self, insightface, image, mask, rotate_mode, pad_top, pad_right, pad_bottom, pad_left):
        """
        Computes the face rotation angle using detected keypoints.

        Args:
            insightface (INSIGHTFACE_APP): The facial recognition model.
            image (IMAGE): The input image containing a face.
            mask (MASK): Mask defining the area of interest.
            rotate_mode (str): The selected rotation mode.
            pad_top (int): Padding applied to the top of the image.
            pad_right (int): Padding applied to the right of the image.
            pad_bottom (int): Padding applied to the bottom of the image.
            pad_left (int): Padding applied to the left of the image.

        Returns:
            tuple: A tuple containing the computed rotation angle.
        """
        # Extract bounding box for the masked face area
        p_x1, p_y1, p_x2, p_y2 = get_mask_bbox_with_padding(mask.squeeze(0), pad_top, pad_right, pad_bottom, pad_left)
        
        # Crop image to the detected face region
        image = image[:, p_y1:p_y2, p_x1:p_x2]
        
        # Extract keypoints from the cropped image
        kps = get_kps_from_image(image, insightface)
        
        angle = 0.0
        if rotate_mode != "none":
            angle = get_angle(
                kps[0], kps[1],
                round_angle=True if rotate_mode == "loseless" else False
            )
        return (angle,)


class ComposeRotated:
    """
    A class responsible for composing a rotated image with the original dimensions.
    Ensures that the rotated image aligns correctly by applying necessary padding adjustments.
    """

    def init(self):
        """
        Initializes the ComposeRotated class.
        """
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "original_image": ("IMAGE",),  # Original input image before rotation
                "rotated_image": ("IMAGE",),  # Rotated image to be aligned
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Output type: composed image
    RETURN_NAMES = ("image",)  # Output name
    FUNCTION = "compose_rotate"  # Function to be executed
    CATEGORY = CATEGORY_NAME  # UI category

    def compose_rotate(self, original_image, rotated_image):
        """
        Adjusts the rotated image to align with the original image's dimensions.
        Applies necessary cropping to match the original image size.

        Args:
            original_image (IMAGE): The original input image before rotation.
            rotated_image (IMAGE): The rotated image to be adjusted.

        Returns:
            tuple: A tuple containing the final aligned image.
        """
        original_width, original_height = original_image.shape[2], original_image.shape[1]
        rotated_width, rotated_height = rotated_image.shape[2], rotated_image.shape[1]

        # Calculate padding adjustments for width
        if rotated_width != original_width:
            pad_x1 = (rotated_width - original_width) // 2
            pad_x2 = pad_x1 * -1
        else:
            pad_x1, pad_x2 = 0, original_width

        # Calculate padding adjustments for height
        if rotated_height != original_height:
            pad_y1 = (rotated_height - original_height) // 2
            pad_y2 = pad_y1 * -1
        else:
            pad_y1, pad_y2 = 0, original_height

        # Crop the rotated image to match the original dimensions
        image = rotated_image[:, pad_y1:pad_y2, pad_x1:pad_x2, :]
        return (image,)


class LoadInstantIdAdapter:
    """
    A class responsible for loading the InstantID adapter and its corresponding resampler.
    """
    
    def init(self):
        """
        Initializes the LoadInstantIdAdapter class.
        """
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter"),
                               {"tooltip": "The default folder where the adapter is searched for is: models/ipadapter."})
            }
        }

    RETURN_TYPES = ("INSTANTID_ADAPTER", "RESAMPLER")  # Output types: InstantID adapter and resampler
    RETURN_NAMES = ("InstantId_adapter", "resampler")  # Output names
    FUNCTION = "load_instantId_adapter"  # Function to be executed
    CATEGORY = CATEGORY_NAME  # UI category

    def load_instantId_adapter(self, ipadapter):
        """
        Loads the InstantID adapter and initializes the resampler.

        Args:
            ipadapter (str): The selected InstantID adapter file.

        Returns:
            tuple: A tuple containing the InstantID adapter instance and its resampler.
        """
        ipadapter_path = folder_paths.get_full_path("ipadapter", ipadapter)
        model = comfy.utils.load_torch_file(ipadapter_path, safe_load=True)
        instant_id = InstantId(model['ip_adapter'])

        resampler = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=16,
            embedding_dim=512,
            output_dim=2048,
            ff_mult=4
        )
        resampler.load_state_dict(model["image_proj"])  
        return (instant_id, resampler)


class InstantIdAdapterApply:
    """
    A class responsible for applying the InstantID adapter to a model,
    modifying its layers using identity embeddings and strength control.
    """
    
    def init(self):
        """
        Initializes the InstantIdAdapterApply class.
        """
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "model": ("MODEL",),  # The base model to apply modifications
                "instantId_adapter": ("INSTANTID_ADAPTER",),  # The InstantID adapter instance
                "face_conditioning": ("FACE_CONDITIONING",),  # Face conditioning embedding
                "strength": ("FLOAT", {"default": 0.8, "min": 0, "step": 0.1, "max": 10}),  # Strength of adaptation
            }
        }

    RETURN_TYPES = ("MODEL",)  # Output type: modified model
    RETURN_NAMES = ("model",)  # Output name
    FUNCTION = "apply_instantId_adapter"  # Function to be executed
    CATEGORY = CATEGORY_NAME  # UI category

    def apply_instantId_adapter(self, model, instantId_adapter, face_conditioning, strength):
        """
        Applies the InstantID adapter to a model, modifying layers based on identity embeddings.

        Args:
            model (MODEL): The base model to apply modifications.
            instantId_adapter (INSTANTID_ADAPTER): The InstantID adapter instance.
            face_conditioning (FACE_CONDITIONING): Embedding used to control identity adaptation.
            strength (float): Strength of the identity adaptation effect.

        Returns:
            tuple: A modified model instance with InstantID applied.
        """
        if strength == 0:
            return model,

        instantId = instantId_adapter.to(comfy.model_management.get_torch_device())
        patch_kwargs = {
            "instantId": instantId,
            "scale": strength,
            "cond": face_conditioning,
            "number": 0
        }

        modified_model = model.clone()

        # Apply patches to input and output layers
        for layer_id in [4, 5, 7, 8]:
            block_indices = range(2) if layer_id in [4, 5] else range(10)
            for index in block_indices:
                set_model_patch_replace(modified_model, patch_kwargs, ("input", layer_id, index))
                patch_kwargs["number"] += 1
            block_indices = range(2) if layer_id in [3, 4, 5] else range(10)
            for index in block_indices:
                set_model_patch_replace(modified_model, patch_kwargs, ("output", layer_id, index))
                patch_kwargs["number"] += 1
        
        # Apply patches to middle layers
        for index in range(10):
            set_model_patch_replace(modified_model, patch_kwargs, ("middle", 1, index))
            patch_kwargs["number"] += 1
        return (modified_model,)


class ControlNetInstantIdApply:
    """
    A class that applies ControlNet conditioning with InstantID face embeddings.
    Based on ControlNetApplyAdvance from ComfyUI/nodes.py.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "positive": ("CONDITIONING",),  # Positive conditioning input
                "negative": ("CONDITIONING",),  # Negative conditioning input
                "face_conditioning": ("FACE_CONDITIONING",),  # Face conditioning embedding
                "control_net": ("CONTROL_NET",),  # ControlNet model instance
                "image": ("IMAGE",),  # Control image for guidance
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})  # Strength of ControlNet effect
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")  # Output types: positive and negative conditioning
    RETURN_NAMES = ("positive", "negative")  # Output names
    FUNCTION = "apply_controlnet"  # Function to be executed
    CATEGORY = CATEGORY_NAME  # UI category

    def apply_controlnet(self, positive, negative, face_conditioning, control_net, image, strength):
        """
        Applies ControlNet with InstantID embeddings to generate conditioned images.

        Args:
            positive (CONDITIONING): Positive conditioning input.
            negative (CONDITIONING): Negative conditioning input.
            face_conditioning (FACE_CONDITIONING): Embedding for identity preservation.
            control_net (CONTROL_NET): ControlNet model instance.
            image (IMAGE): Image used as guidance.
            strength (float): Strength of ControlNet's influence.

        Returns:
            tuple: Processed positive and negative conditionings with ControlNet applied.
        """
        if strength == 0:
            return positive, negative

        control_hint = image.movedim(-1, 1)  # Adjust image dimensions for processing
        cnets = {}
        out = []

        for conditioning, is_positive in zip([positive, negative], [True, False]):
            conditioned_output = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get("control", None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                if is_positive:
                    d["cross_attn_controlnet"] = face_conditioning.to(comfy.model_management.intermediate_device())
                else:
                    d["cross_attn_controlnet"] = torch.zeros_like(face_conditioning).to(comfy.model_management.intermediate_device())
                
                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                conditioned_output.append([t[0], d])
            
            out.append(conditioned_output) 
        return (out[0], out[1],)


class InstantIdAndControlnetApply:
    """
    A class responsible for applying InstantID and ControlNet together to a model.
    This allows enhanced face swapping by leveraging identity embeddings and control images.
    """

    def init(self):
        """
        Initializes the InstantIdAndControlnetApply class.
        """
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "model": ("MODEL",),  # The base model used for processing
                "ipadapter_path": (folder_paths.get_filename_list("ipadapter"), {"tooltip": "The default folder where the adapter is searched for is: models/ipadapter."}),  # Path to the InstantID adapter
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),  # Name of the ControlNet model
                "face_embed": ("FACE_EMBED",),  # Face embedding for identity preservation
                "control_image": ("IMAGE",),  # Control image for ControlNet
                "adapter_strength": ("FLOAT", {"default": 0.5, "min": 0, "step": 0.1, "max": 10}),  # Strength of InstantID effect
                "control_net_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01}),  # Strength of ControlNet effect
                "positive": ("CONDITIONING",),  # Positive conditioning prompts
                "negative": ("CONDITIONING",)  # Negative conditioning prompts
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")  # Output types: modified model and conditionings
    RETURN_NAMES = ("model", "positive", "negative")  # Output names
    FUNCTION = "apply_instantId_adapter_and_controlnet"  # Function to be executed
    CATEGORY = CATEGORY_NAME  # UI category

    def apply_instantId_adapter_and_controlnet(self, model, ipadapter_path, control_net_name, face_embed, control_image,
                                               adapter_strength, control_net_strength, positive, negative):
        """
        Applies the InstantID adapter and ControlNet to modify the model and conditionings.

        Args:
            model (MODEL): The base model to apply transformations.
            ipadapter_path (str): Path to the InstantID adapter.
            control_net_name (str): Name of the ControlNet model.
            face_embed (FACE_EMBED): Face embedding to guide identity preservation.
            control_image (IMAGE): Image used as a control reference.
            adapter_strength (float): Strength of the InstantID effect.
            control_net_strength (float): Strength of the ControlNet effect.
            positive (CONDITIONING): Positive conditioning inputs.
            negative (CONDITIONING): Negative conditioning inputs.
        Returns:
            dict: A dictionary containing modified model and conditionings.
        """
        graph = GraphBuilder()
        
        # Load InstantID adapter
        load_instant_id_adapter = graph.node(
            "LoadInstantIdAdapter", ipadapter=ipadapter_path
        )
        
        # Combine face embeddings
        face_embed_combine = graph.node(
            "FaceEmbedCombine", resampler=load_instant_id_adapter.out(1), face_embeds=face_embed
        )
        
        # Load ControlNet model
        load_control_net = graph.node(
            "ControlNetLoader", control_net_name=control_net_name
        )
        
        # Apply InstantID adapter to modify model
        instant_id_apply = graph.node(
            "InstantIdAdapterApply", model=model, instantId_adapter=load_instant_id_adapter.out(0),
            face_conditioning=face_embed_combine.out(0), strength=adapter_strength
        )
        
        # Apply ControlNet and combine it with InstantID effect
        controlnet_instantid_apply = graph.node(
            "ControlNetInstantIdApply", positive=positive, negative=negative,
            face_conditioning=face_embed_combine.out(0), control_net=load_control_net.out(0),
            image=control_image, strength=control_net_strength
        )
        return {
            "result": (instant_id_apply.out(0), controlnet_instantid_apply.out(0), controlnet_instantid_apply.out(1),),
            "expand": graph.finalize()
        }


class PreprocessImageAdvanced:
    """
    A class responsible for advanced image preprocessing, including resizing, padding,
    and facial keypoint extraction for face swap applications.
    """
    resize_modes = ["auto", "free", "scale by width", "scale by height"]
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    def init(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required and optional input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Pose image."}),
                "width": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
                "resize_mode": (cls.resize_modes,),
                "upscale_method": (cls.upscale_methods,),
                "pad_top": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
                "pad_right": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
                "pad_bottom": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
                "pad_left": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),
            },
            "optional": {
                "insightface": ("INSIGHTFACE_APP",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("resized_image", "mask", "control_image", "x", "y", "original_width", "original_height", "new_width", "new_height")
    FUNCTION = "preprocess_image"
    CATEGORY = CATEGORY_NAME
  
    def preprocess_image(self, image, width, height, resize_mode, upscale_method, pad_top, pad_right, pad_bottom, pad_left, insightface=None):
        """
        Processes an image by resizing, padding, and applying facial detection.

        Args:
            image (IMAGE): The input image to be processed.
            width (int): Desired width of the output image.
            height (int): Desired height of the output image.
            resize_mode (str): The mode used for resizing the image.
            upscale_method (str): The interpolation method used for upscaling.
            pad_top (int): Padding applied to the top of the image.
            pad_right (int): Padding applied to the right of the image.
            pad_bottom (int): Padding applied to the bottom of the image.
            pad_left (int): Padding applied to the left of the image.
            insightface (Optional[INSIGHTFACE_APP]): Optional facial recognition model.

        Returns:
            tuple: Processed image, mask, control image (if applicable), and bounding box coordinates.
        """
        face_image = (255.0 * image.cpu().numpy().squeeze()).clip(0, 255).astype(np.uint8)
        face_info = insightface.get(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        assert len(face_info) > 0, "No face detected for preprocess image"
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        face_bbox = face_info["bbox"]
    
        _, original_height, original_width, _ = image.shape
        new_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        x_min, y_min, x_max, y_max = [int(x) for x in face_bbox]
        new_mask[y_min:y_max, x_min:x_max] = 1
        new_mask = torch.from_numpy(new_mask).to(dtype=torch.float32)
        p_x1, p_y1, p_x2, p_y2 = get_mask_bbox_with_padding(new_mask, pad_top, pad_right, pad_bottom, pad_left)
        mask = new_mask.unsqueeze(0)
        mask = mask[:, p_y1:p_y2, p_x1:p_x2]
        image = image[:, p_y1:p_y2, p_x1:p_x2]
        kps = get_kps_from_image(image, insightface) if insightface else None
        _, original_height, original_width, _ = image.shape
        
        if resize_mode == "auto":
            width, height = resize_to_fit_area(int(p_x2 - p_x1), int(p_y2 - p_y1), width, height)
        elif resize_mode != "free":
            ratio = original_width / original_height
            width, height = (int(height * ratio), height) if resize_mode == "scale by height" else (width, int(width / ratio))
        
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        image = image.movedim(-1, 1)
        mask = mask.movedim(-1, 1)
        
        mask = comfy.utils.common_upscale(mask, width, height, "bilinear", "disabled")
        image = comfy.utils.common_upscale(image, width, height, upscale_method, "disabled")
        
        mask = mask.movedim(1, -1)[:, :, :, 0]
        image = image.movedim(1, -1)
        _, new_height, new_width = mask.shape
        
        control_image = None
        if kps is not None:
            kps *= [image.shape[2] / original_width, image.shape[1] / original_height]
            control_image = draw_kps(width, height, kps)
            control_image = (torch.from_numpy(control_image).float() / 255.0).unsqueeze(0)
        return (image, mask, control_image, p_x1, p_y1, original_width, original_height, new_width, new_height)


class PreprocessImage(PreprocessImageAdvanced):
    """
    A class responsible for simple preprocessing of an image, including resizing and padding.
    Inherits from PreprocessImageAdvanced.
    """

    def init(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required and optional input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "image": ("IMAGE",),  # Input image to be processed
                "width": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),  # Desired image width
                "height": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),  # Desired image height
                "resize_mode": (cls.resize_modes,),  # Resizing mode
                "pad": ("INT", {"default": 100, "min": 0, "step": 1, "max": MAX_RESOLUTION}),  # Padding value
            },
            "optional": {
                "insightface": ("INSIGHTFACE_APP",),  # Optional facial recognition model
            }
        }

    FUNCTION = "preprocess_image_simple"  # Function to be executed
    CATEGORY = CATEGORY_NAME  # UI category

    def preprocess_image_simple(self, image, width, height, resize_mode, pad, insightface=None):
        """
        Performs a simple preprocessing operation by resizing and padding the image.
        
        Args:
            image (IMAGE): The input image to be processed.
            width (int): Desired width of the output image.
            height (int): Desired height of the output image.
            resize_mode (str): The mode used for resizing the image.
            pad (int): Padding size applied to all sides of the image.
            insightface (Optional[INSIGHTFACE_APP]): Optional facial recognition model.

        Returns:
            tuple: Processed image with applied transformations.
        """
        return self.preprocess_image(
            image, width, height, resize_mode, "bilinear", pad, pad, pad, pad, insightface
        )


class LoadInsightface:
    """
    A class responsible for loading the InsightFace model for facial analysis.
    """

    def init(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required input parameters for the function.

        Returns:
            dict: A dictionary specifying input types (empty in this case).
        """
        return {}

    RETURN_TYPES = ("INSIGHTFACE_APP",)  # Output type: InsightFace application instance
    RETURN_NAMES = ("insightface",)  # Output name
    FUNCTION = "load_insightface"  # Function to be called
    CATEGORY = CATEGORY_NAME  # UI category

    def load_insightface(self):
        """
        Loads and prepares the InsightFace application for facial recognition and analysis.

        Returns:
            tuple: A tuple containing the initialized InsightFace application instance.
        """
        app = FaceAnalysis(
            name="antelopev2",
            root=INSIGHTFACE_PATH,
            providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640))
        return (app,)


class KpsMaker:
    """
    A class responsible for generating control images based on face keypoints and masks for face swapping.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required and optional input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and their properties.
        """
        return {
            "required": {
                "image": ("STRING",),  # Name of the input image file
                "width": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),  # Image width
                "height": ("INT", {"default": 1024, "min": 0, "step": 1, "max": MAX_RESOLUTION}),  # Image height
            },
            "optional": {
                "image_reference": ("IMAGE",),  # Optional reference image for comparison
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)  # Output types: processed image and mask
    RETURN_NAMES = ("control_image", "mask",)  # Output names
    FUNCTION = "draw_kps"  # Function to be executed
    CATEGORY = CATEGORY_NAME  # UI category

    def draw_kps(self, image):
        """
        Processes the input image and generates a control image along with a mask.

        Args:
            image (str): Name of the input image file.

        Returns:
            tuple: A tuple containing the control image (torch.Tensor) and the generated mask (torch.Tensor).
        """
        if "clipspace" not in image:
            image_path = os.path.join(
                folder_paths.get_input_directory(), "faceswap_controls", image
            )
        else:  # With mask - saved in a different directory
            image_path = os.path.join(
                folder_paths.get_input_directory(), image[:-8]
            )

        # Load and convert the image
        pil_image = node_helpers.pillow(Image.open, image_path)
        image = pil_image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        # Generate mask if an alpha channel exists, otherwise create an empty mask
        if "A" in pil_image.getbands():
            mask = np.array(pil_image.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        mask = mask.unsqueeze(0)
        return (image, mask)


class RotateImage:
    """
    A class for rotating an image with optional padding to preserve its original content.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the required input parameters for the function.

        Returns:
            dict: A dictionary specifying input types and default values.
        """
        return {
            "required": {
                "image": ("IMAGE",),  # Input image to be rotated
                "angle": ("FLOAT", {"default": 0.0, "min": -360.0, "step": 0.1, "max": 360.0}),  # Rotation angle in degrees
                "counter_clockwise": ("BOOLEAN", {"default": True}),  # Direction of rotation
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Output type: rotated image
    RETURN_NAMES = ("rotated_image", "rotated_mask",)  # Output names
    FUNCTION = "rotate_and_pad_image"  # Function to be called
    CATEGORY = CATEGORY_NAME  # Category for UI organization

    def rotate_and_pad_image(self, image, angle, counter_clockwise):
        """
        Rotates the given image by the specified angle. If the angle is 0 or 360,
        the function returns the original image unchanged.

        Args:
            image (IMAGE): The input image to be rotated.
            angle (float): The rotation angle in degrees.
            counter_clockwise (bool): If True, rotates counter-clockwise, else rotates clockwise.

        Returns:
            tuple: A tuple containing the rotated image.
        """
        if angle == 0 or angle == 360:
            return image,

        image = rotate_with_pad(image, counter_clockwise, angle)
        return (image,)


NODE_CLASS_MAPPINGS = {
  "LoadInsightface": LoadInsightface,
  "LoadInstantIdAdapter": LoadInstantIdAdapter,
  "InstantIdAdapterApply": InstantIdAdapterApply,
  "ControlNetInstantIdApply": ControlNetInstantIdApply,
  "InstantIdAndControlnetApply": InstantIdAndControlnetApply,
  "PreprocessImage": PreprocessImage,
  "PreprocessImageAdvanced": PreprocessImageAdvanced,
  "AngleFromFace": AngleFromFace,
  "RotateImage": RotateImage,
  "ComposeRotated": ComposeRotated,
  "KpsMaker": KpsMaker,
  "FaceEmbed": FaceEmbed,
  "FaceEmbedCombine": FaceEmbedCombine
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "LoadInsightface": "Load insightface",
  "LoadInstantIdAdapter": "Load instantId adapter",
  "InstantIdAdapterApply": "Apply instantId adapter",
  "ControlNetInstantIdApply": "Apply instantId ControlNet",
  "InstantIdAndControlnetApply": "Apply instantId and ControlNet",
  "PreprocessImage": "Preprocess image for instantId",
  "PreprocessImagAdvancese": "Preprocess image for instantId (Advanced)",
  "AngleFromFace": "Get Angle from face",
  "RotateImage": "Rotate Image",
  "ComposeRotated": "Remove rotation padding",
  "KpsMaker": "Draw KPS",
  "FaceEmbed": "FaceEmbed for instantId",
  "FaceEmbedCombine": "FaceEmbed Combine"
}