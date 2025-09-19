import zlib
import torch
import math
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

def get_passwd(password: str):
    if not (isinstance(password, str) and len(password) == 4 and password.isdigit()):
        raise ValueError("Password must be a four-digit numeric string")
    step = int(password[0:2])
    v = int(password[2:3])
    h = int(password[3:4])
    return max(1, step), v, h

def _generate2d(x, y, ax, ay, bx, by, coordinates):
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = int(np.sign(ax)), int(np.sign(ay))
    dbx, dby = int(np.sign(bx)), int(np.sign(by))
    if h == 1:
        for _ in range(w):
            coordinates.append((x, y)); x += dax; y += day
        return
    if w == 1:
        for _ in range(h):
            coordinates.append((x, y)); x += dbx; y += dby
        return
    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2, h2 = abs(ax2 + ay2), abs(bx2 + by2)
    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2): ax2 += dax; ay2 += day
        _generate2d(x, y, ax2, ay2, bx, by, coordinates)
        _generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, coordinates)
    else:
        if (h2 % 2) and (h > 2): bx2 += dbx; by2 += dby
        _generate2d(x, y, bx2, by2, ax2, ay2, coordinates)
        _generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, coordinates)
        _generate2d(x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby), -bx2, -by2, -(ax - ax2), -(ay - ay2), coordinates)
        
def gilbert2d(width, height):
    coordinates = []
    if width >= height: _generate2d(0, 0, width, 0, 0, height, coordinates)
    else: _generate2d(0, 0, 0, height, width, 0, coordinates)
    return coordinates

def add_padding(original_data, original_width, original_height, extra_cols, extra_rows):
    new_width, new_height = original_width + extra_cols, original_height + extra_rows
    padded_data = np.zeros((new_height, new_width, 4), dtype=np.uint8)
    padded_data[:original_height, :original_width] = original_data
    if extra_cols > 0:
        last_col = original_data[:, -1, :]
        padded_data[:original_height, original_width:] = last_col[:, np.newaxis, :]
    if extra_rows > 0:
        last_row = padded_data[original_height - 1, :, :]
        padded_data[original_height:, :] = last_row[np.newaxis, :, :]
    return padded_data

def crop_image_data(image_data, remove_columns, remove_rows):
    original_height, original_width, _ = image_data.shape
    return image_data[:original_height - remove_rows, :original_width - remove_columns, :]

def calculate_final_mapping(initial_map, steps):
    size = len(initial_map)
    final_map = np.arange(size, dtype=np.int64)
    current_power_map = initial_map
    while steps > 0:
        if steps % 2 == 1:
            final_map = current_power_map[final_map]
        current_power_map = current_power_map[current_power_map]
        steps //= 2
    return final_map

def encrypt_data(img_data, password):
    step, v, h = get_passwd(password)
    height, width, _ = img_data.shape
    data = img_data.reshape(-1, 4)
    total_pixels = width * height
    
    curve_np = np.array(gilbert2d(width, height), dtype=np.int64)
    old_indices = curve_np[:, 1] * width + curve_np[:, 0]
    
    offset_float = ((math.sqrt(5) - 1) / 2) * total_pixels
    offset = int(math.floor(offset_float + 0.5))
    
    new_pos_indices_on_curve = (np.arange(total_pixels, dtype=np.int64) + offset) % total_pixels
    
    initial_map = np.empty_like(old_indices)
    initial_map[old_indices[new_pos_indices_on_curve]] = old_indices
    
    final_map = calculate_final_mapping(initial_map, step)
    scrambled_flat_data = data[final_map]
    
    scrambled_data = scrambled_flat_data.reshape(height, width, 4)
    padded_data = add_padding(scrambled_data, width, height, v, h)
    return padded_data

def decrypt_data(img_data, password):
    step, v, h = get_passwd(password)
    
    cropped_data = crop_image_data(img_data, v, h)
    height, width, _ = cropped_data.shape
    data = cropped_data.reshape(-1, 4)
    total_pixels = width * height
    
    curve_np = np.array(gilbert2d(width, height), dtype=np.int64)
    old_indices = curve_np[:, 1] * width + curve_np[:, 0]
    
    offset_float = ((math.sqrt(5) - 1) / 2) * total_pixels
    offset = int(math.floor(offset_float + 0.5))
    
    new_pos_indices_on_curve = (np.arange(total_pixels, dtype=np.int64) + offset) % total_pixels
    
    initial_map = np.empty_like(old_indices)
    initial_map[old_indices] = old_indices[new_pos_indices_on_curve]
    
    final_map = calculate_final_mapping(initial_map, step)
    unscrambled_flat_data = data[final_map]
    
    unscrambled_data = unscrambled_flat_data.reshape(height, width, 4)
    return unscrambled_data

class EncryptDecryptImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "image": ("IMAGE", ),
                "mode": (["encrypt", "decrypt"], {"default": "encrypt"}),
                "password": ("STRING", ),
                }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    
    CATEGORY = "image"
    
    def main(self, image, mode, password):
        results = []
        images = tensor2pil(image)
        password = password if password != "" else "0000"
        for image in images:
            results.append(process_pil_image(mode, image, password))
        return (array2tensor(results), )

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
        result = bwm.init_block_add_index(input.shape)
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
                    "min": -1,
                    "step": 1
                }),
                "original_height": ("INT", {
                    "default": -1,
                    "min": -1,
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
                    "max": 98,
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
    "EncryptDecryptImage": EncryptDecryptImage,
    "ApplyBlindWatermark": ApplyBlindWatermark,
    "ApplyBlindWatermarkAdvanced": ApplyBlindWatermarkAdvanced,
    "DecodeBlindWatermark": DecodeBlindWatermark,
    "DecodeBlindWatermarkAdvanced": DecodeBlindWatermarkAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EncryptDecryptImage": "Encrypt/Decrypt Image",
    "ApplyBlindWatermark": "Apply Blind Watermark",
    "ApplyBlindWatermarkAdvanced": "Apply Blind Watermark (Advanced)",
    "DecodeBlindWatermark": "Decode Blind Watermark",
    "DecodeBlindWatermarkAdvanced": "Decode Blind Watermark (Advanced)"
}