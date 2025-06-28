import zlib
import torch
import numpy as np
from PIL import Image
from .BlindWatermark import watermark

def tensor2pil(image):
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out
    return [Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))]

def pil2tensor(image):
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def array2tensor(image):
    if isinstance(image, list):
        return torch.cat([array2tensor(img) for img in image], dim=0)
    return torch.from_numpy(np.round(image).astype(np.float32) / 255.0).unsqueeze(0)

def number_hash(data, length=8):
    crc = zlib.crc32(str(data).encode())
    return str(crc % (10 ** length)).zfill(length)

class ApplyBlindWatermark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "watermark_image": ("IMAGE",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624,
                    "step": 1
                })
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_blind_watermark"
    CATEGORY = "BlindWatermark/encode"
    DESCRIPTION = "Embeds an invisible watermark in the input image.\n(use advanced nodes for more options)"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True
    
    def apply_blind_watermark(self, original_image, watermark_image, seed):
        num = 1
        results = []
        seed_hash = number_hash(seed)
        seed_a = int(str(seed_hash)[:4])
        seed_b = int(str(seed_hash)[-4:])
        wm_img = np.array(tensor2pil(watermark_image)[0].resize((64, 64)).convert("RGB"))
        print("[BlindWatermark] Watermark loaded.")
        
        bwm = watermark(seed_a, seed_b, 30, block_shape=(6, 6), dwt_deep=1)
        original_images = tensor2pil(original_image)
        for img in original_images:
            print(f"[BlindWatermark] Embedding image {num}/{len(original_image)} ...")
            bwm.read_ori_img(np.array(img)[:, :, ::-1])
            result = bwm.read_wm(wm_img[:, :, ::-1])
            if not result:
                raise RuntimeError("Unable to embed watermark, input image size is too small!")
            results.append(bwm.embed()[:, :, ::-1])
            num += 1
        print(f"[BlindWatermark] All done.")
        return (array2tensor(results),)

class ApplyBlindWatermarkAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "watermark_image": ("IMAGE",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624,
                    "step": 1
                }),
                "watermark_size": ("INT", {
                    "default": 64,
                    "min": 16,
                    "step": 2
                }),
                "strength": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 99,
                    "step": 1
                }),
                "block_size": (["2", "4", "6", "8"], {
                    "default": "6"
                }),
                "robustness": (["1", "2", "3"], {
                    "default": "1"
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_blind_watermark"
    CATEGORY = "BlindWatermark/encode"
    DESCRIPTION = """strength:    larger = more robust (but also more artifacts)
block_size: larger = more invisible (but image may not have enough space)
robustness: larger = more robust (but image may not have enough space)"""
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True

    def apply_blind_watermark(self, original_image, watermark_image, seed, watermark_size, strength, block_size, robustness):
        num = 1
        results = []
        seed_hash = number_hash(seed)
        seed_a = int(str(seed_hash)[:4])
        seed_b = int(str(seed_hash)[-4:])
        wm_img = np.array(tensor2pil(watermark_image)[0].resize((watermark_size, watermark_size)).convert("RGB"))
        print("[BlindWatermark] Watermark loaded.")
        
        bwm = watermark(seed_a, seed_b, strength, block_shape=(int(block_size),int(block_size)), dwt_deep=int(robustness))
        original_images = tensor2pil(original_image)
        for img in original_images:
            print(f"[BlindWatermark] Embedding image {num}/{len(original_image)} ...")
            bwm.read_ori_img(np.array(img)[:, :, ::-1])
            result = bwm.read_wm(wm_img[:, :, ::-1])
            if not result:
                raise RuntimeError("Unable to embed watermark! Try reducing \"watermark_size\" or \"block_size\" or \"robustness\"!")
            results.append(bwm.embed()[:, :, ::-1])
            num += 1
        print(f"[BlindWatermark] All done.")
        return (array2tensor(results),)

class DecodeBlindWatermark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "original_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624,
                    "step": 1
                })
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("watermark",)
    FUNCTION = "decode_blind_watermark"
    CATEGORY = "BlindWatermark/decode"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True
    
    def decode_blind_watermark(self, image, original_seed):
        seed_hash = number_hash(original_seed)
        seed_a = int(str(seed_hash)[:4])
        seed_b = int(str(seed_hash)[-4:])
        input = tensor2pil(image)[0]
        input = np.array(input)[:, :, ::-1]
        bwm = watermark(seed_a, seed_b, 30, wm_shape=(64, 64), block_shape=(6, 6), dwt_deep=1)
        result = bwm.init_block_add_index(image.shape)
        if not result:
            raise RuntimeError("The size of the watermark exceeds the image capacity!")
        print("[BlindWatermark] Extracting watermark...")
        wm_imgs = bwm.extract(input, channel="YUV", invert="")
        print("[BlindWatermark] Done.")
        img = Image.fromarray(np.uint8(wm_imgs[-1])).convert("RGB")
        return (pil2tensor([img]),)

class DecodeBlindWatermarkAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "original_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624,
                    "step": 1
                }),
                "original_width": ("INT", {
                    "default": -1,
                    "step": 1
                }),
                "original_height": ("INT", {
                    "default": -1,
                    "step": 1
                }),
                "watermark_size": ("INT", {
                    "default": 64,
                    "min": 16,
                    "step": 2
                }),
                "strength": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 99,
                    "step": 1
                }),
                "block_size": (["2", "4", "6", "8"], {
                    "default": "6"
                }),
                "robustness": (["1", "2", "3"], {
                    "default": "1"
                }),
                "y_channel": (["on","off","invert"], {
                    "default": "on"
                }),
                "u_channel": (["on","off","invert"], {
                    "default": "on"
                }),
                "v_channel": (["on","off","invert"], {
                    "default": "on"
                })
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("watermark",)
    FUNCTION = "decode_blind_watermark"
    CATEGORY = "BlindWatermark/decode"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True
    
    def decode_blind_watermark(self, image, original_seed, original_width, original_height, watermark_size, strength, block_size, robustness, y_channel, u_channel, v_channel):
        channels = []
        inverts = []
        if y_channel in ["on", "invert"]:
            channels.append("Y")
            if y_channel == "invert":
                inverts.append("Y")
        if u_channel in ["on", "invert"]:
            channels.append("U")
            if u_channel == "invert":
                inverts.append("U")
        if v_channel in ["on", "invert"]:
            channels.append("V")
            if v_channel == "invert":
                inverts.append("V")
        if "Y" not in channels and "U" not in channels and "V" not in channels:
            raise RuntimeError("One of the Y/U/V channels is required!")
        
        seed_hash = number_hash(original_seed)
        seed_a = int(str(seed_hash)[:4])
        seed_b = int(str(seed_hash)[-4:])
        input = tensor2pil(image)[0]
        if original_width != -1 and original_height != -1:
            original_size = (original_width, original_height)
            if input.size != original_size:
                input = input.resize(original_size)
        
        input = np.array(input)[:, :, ::-1]
        bwm = watermark(seed_a, seed_b, strength, wm_shape=(watermark_size, watermark_size), block_shape=(int(block_size),int(block_size)), dwt_deep=int(robustness))
        result = bwm.init_block_add_index(input.shape)
        if not result:
            raise RuntimeError("The size of the watermark exceeds the image capacity!")
        print(f"[BlindWatermark] Extracting watermark from channel {'/'.join(channels)}...")
        wm_imgs = bwm.extract(input, channel="".join(channels), invert="".join(inverts))
        img = Image.fromarray(np.uint8(wm_imgs[-1])).convert("RGB")
        print("[BlindWatermark] Done.")
        return (pil2tensor([img]),)

NODE_CLASS_MAPPINGS = {
    "ApplyBlindWatermark": ApplyBlindWatermark,
    "ApplyBlindWatermarkAdvanced": ApplyBlindWatermarkAdvanced,
    "DecodeBlindWatermark": DecodeBlindWatermark,
    "DecodeBlindWatermarkAdvanced": DecodeBlindWatermarkAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyBlindWatermark": "Apply Blind Watermark",
    "ApplyBlindWatermarkAdvanced": "Apply Blind Watermark (Advanced)",
    "DecodeBlindWatermark": "Decode Blind Watermark",
    "DecodeBlindWatermarkAdvanced": "Decode Blind Watermark (Advanced)"
}