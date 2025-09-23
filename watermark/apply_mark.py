import numpy as np
from PIL import Image
import pywt
from dataclasses import dataclass

@dataclass
class WatermarkResult:
    watermarked_rgb: Image.Image
    wm_YCbCr: tuple
    Y_orig_dwt: tuple
    Y_wm_dwt: tuple

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32)

def np_to_pil(arr: np.ndarray, mode="RGB") -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=mode)

def rgb_to_ycbcr_np(rgb: np.ndarray) -> np.ndarray:
    transform = np.array([
        [ 0.299,     0.587,     0.114   ],
        [-0.168736, -0.331264,  0.5     ],
        [ 0.5,      -0.418688, -0.081312]
    ], dtype=np.float32)
    shift = np.array([0, 128, 128], dtype=np.float32)
    return rgb @ transform.T + shift

def ycbcr_to_rgb_np(ycbcr: np.ndarray) -> np.ndarray:
    inv = np.array([
        [1.0,  0.0,        1.402     ],
        [1.0, -0.344136,  -0.714136  ],
        [1.0,  1.772,      0.0       ]
    ], dtype=np.float32)
    shift = np.array([0, 128, 128], dtype=np.float32)
    return (ycbcr - shift) @ inv.T

def resize_gray_np(img_gray: np.ndarray, size_hw: tuple) -> np.ndarray:
    h, w = size_hw
    # 修复：移除deprecated的mode参数
    pil_img = Image.fromarray(np.clip(img_gray, 0, 255).astype(np.uint8))
    pil_img = pil_img.resize((w, h), Image.LANCZOS)
    return np.asarray(pil_img).astype(np.float32)

def tile_watermark(wm_gray: np.ndarray, target_shape: tuple,
                   repeat_h: int, repeat_v: int) -> np.ndarray:
    H_target, W_target = target_shape
    # 每个"单元格"的尺寸（向下取整）
    h_unit, w_unit = max(1, H_target // repeat_v), max(1, W_target // repeat_h)
    wm_unit = resize_gray_np(wm_gray, (h_unit, w_unit))
    wm_tiled = np.tile(wm_unit, (repeat_v, repeat_h))
    
    # 修复：确保输出尺寸完全匹配目标
    if wm_tiled.shape != target_shape:
        # 如果平铺后尺寸不匹配，强制调整到目标尺寸
        if wm_tiled.shape[0] >= H_target and wm_tiled.shape[1] >= W_target:
            # 如果平铺结果过大，裁剪
            wm_tiled = wm_tiled[:H_target, :W_target]
        else:
            # 如果平铺结果过小，填充
            pad_h = max(0, H_target - wm_tiled.shape[0])
            pad_w = max(0, W_target - wm_tiled.shape[1])
            wm_tiled = np.pad(wm_tiled, ((0, pad_h), (0, pad_w)), mode='edge')[:H_target, :W_target]
    
    return wm_tiled

def prepare_mask(mask_img: Image.Image, target_shape: tuple,
                 threshold: float | None = None, invert: bool = False) -> np.ndarray:
    """
    将掩膜图转换为与 LL 同尺寸的 [0,1] 浮点掩膜。
    - threshold=None: 软掩膜（灰度/半透明会保留）
    - threshold in [0,255]: 二值化为硬掩膜
    - invert=True: 反相（把保留/屏蔽互换）
    """
    mask_gray = pil_to_np(mask_img.convert("L"))
    mask_r = resize_gray_np(mask_gray, target_shape)  # 与 LL 同尺寸
    
    # 修复：确保掩膜尺寸完全匹配
    H_target, W_target = target_shape
    if mask_r.shape != (H_target, W_target):
        # 强制调整到精确尺寸
        mask_r = mask_r[:H_target, :W_target] if mask_r.shape[0] >= H_target and mask_r.shape[1] >= W_target else np.pad(mask_r, ((0, max(0, H_target-mask_r.shape[0])), (0, max(0, W_target-mask_r.shape[1]))), mode='edge')
    
    if threshold is not None:
        mask_bin = (mask_r >= float(threshold)).astype(np.float32)
        mask_f = mask_bin
    else:
        mask_f = mask_r / 255.0  # 软掩膜
    if invert:
        mask_f = 1.0 - mask_f
    # 保证范围 [0,1]
    return np.clip(mask_f, 0.0, 1.0)

# --------- 带 mask 的平铺嵌入 ---------
def embed_watermark_repeat_mask(
    cover_img: Image.Image,
    wm_img: Image.Image,
    alpha: float = 8.0,
    repeat_h: int = 1,
    repeat_v: int = 1,
    mask_img: Image.Image | None = None,
    mask_threshold: float | None = None,
    mask_invert: bool = False,
) -> WatermarkResult:
    """
    在 Y 通道的一层 DWT 的 LL 子带中嵌入"平铺水印"，并用 mask 限制区域。
    - mask_img: 任意尺寸灰度/透明图；会被缩放到 LL 尺寸。
    - mask_threshold=None: 使用软掩膜；否则按阈值(0~255)二值化。
    - mask_invert: True 则把掩膜反相（在 mask=0 的地方嵌入）。
    """
    # 1) 转 Y 通道
    cover_rgb = pil_to_np(cover_img.convert("RGB"))
    cover_ycbcr = rgb_to_ycbcr_np(cover_rgb)
    Y = cover_ycbcr[..., 0]

    # 2) 一层 DWT
    coeffs = pywt.dwt2(Y, 'haar')
    LL, (LH, HL, HH) = coeffs

    # 3) 平铺水印到 LL 尺寸，并归一化到 [-0.5, 0.5]
    wm_gray = pil_to_np(wm_img.convert("L"))
    wm_tiled = tile_watermark(wm_gray, LL.shape, repeat_h, repeat_v)
    wm_norm = (wm_tiled / 255.0) - 0.5

    # 4) 准备掩膜（与 LL 同尺寸，范围[0,1]）
    if mask_img is not None:
        M = prepare_mask(mask_img, LL.shape, threshold=mask_threshold, invert=mask_invert)
    else:
        M = np.ones_like(LL, dtype=np.float32)  # 不限区域

    # 调试信息（可选）
    print(f"LL shape: {LL.shape}")
    print(f"wm_norm shape: {wm_norm.shape}")
    print(f"M shape: {M.shape}")

    # 5) 区域嵌入：仅在 M>0 的地方生效；支持软掩膜强度渐变
    #    LL' = LL + (alpha * M) * wm_norm
    LL_w = LL + (alpha * M) * wm_norm

    # 6) 逆 DWT 并回到 RGB
    Y_w = pywt.idwt2((LL_w, (LH, HL, HH)), 'haar')
    ycbcr_w = cover_ycbcr.copy()
    ycbcr_w[..., 0] = Y_w
    rgb_w = ycbcr_to_rgb_np(ycbcr_w)

    return WatermarkResult(
        watermarked_rgb=np_to_pil(rgb_w, "RGB"),
        wm_YCbCr=(Y_w, ycbcr_w[..., 1], ycbcr_w[..., 2]),
        Y_orig_dwt=coeffs,
        Y_wm_dwt=(LL_w, (LH, HL, HH))
    )

# --------- 提取（可选使用相同 mask 只看指定区域） ---------
def extract_watermark_repeat_mask(
    cover_img: Image.Image,
    watermarked_img: Image.Image,
    alpha: float,
    repeat_h: int,
    repeat_v: int,
    mask_img: Image.Image | None = None,
    mask_threshold: float | None = None,
    mask_invert: bool = False,
    return_masked_only: bool = False,
) -> Image.Image:
    """
    提取平铺水印： wm_est ≈ (LL_wm - LL_orig) / alpha
    - return_masked_only=True: 只显示掩膜区域（其余置黑）
    """
    cov_rgb = pil_to_np(cover_img.convert("RGB"))
    cov_ycbcr = rgb_to_ycbcr_np(cov_rgb)
    Y_cov = cov_ycbcr[..., 0]
    LL_cov, _ = pywt.dwt2(Y_cov, 'haar')

    wm_rgb = pil_to_np(watermarked_img.convert("RGB"))
    wm_ycbcr = rgb_to_ycbcr_np(wm_rgb)
    Y_wm = wm_ycbcr[..., 0]
    LL_wm, _ = pywt.dwt2(Y_wm, 'haar')

    # 基本提取
    wm_est = (LL_wm - LL_cov) / (alpha + 1e-8)   # 约在 [-0.5, 0.5]
    wm_est = (wm_est + 0.5) * 255.0              # 回到 [0,255]
    wm_est = np.clip(wm_est, 0, 255).astype(np.uint8)

    if mask_img is not None:
        M = prepare_mask(mask_img, LL_cov.shape, threshold=mask_threshold, invert=mask_invert)
        if return_masked_only:
            out = (wm_est.astype(np.float32) * M).astype(np.uint8)
            return Image.fromarray(out)
        # 否则正常返回完整估计图（便于观察 mask 外"无嵌入"的区域也会呈现低能量纹理）
    return Image.fromarray(wm_est)


if __name__ == "__main__":
    # 检查文件是否存在
    try:
        cover = Image.open("cover.jpg")
        print("载体图像加载成功")
    except FileNotFoundError:
        print("错误：找不到 cover.jpg 文件")
        exit(1)
    
    try:
        wm = Image.open("watermark.png")
        print("水印图像加载成功")
    except FileNotFoundError:
        print("错误：找不到 watermark.png 文件")
        exit(1)
    
    try:
        mask = Image.open("mask.png")
        print("掩膜图像加载成功")
    except FileNotFoundError:
        print("警告：找不到 mask.png 文件，将不使用掩膜")
        mask = None

    # 例：水印在水平重复 6 次、垂直重复 4 次；只在 mask 指定区域出现
    alpha = 25.0
    result = embed_watermark_repeat_mask(
        cover_img=cover,
        wm_img=wm,
        alpha=alpha,
        repeat_h=2,
        repeat_v=2,
        mask_img=mask,
        mask_threshold=None,   # None=软掩膜；如想硬边界可设阈值 128
        mask_invert=False      # True 则反相：黑区嵌入、白区不嵌入
    )
    result.watermarked_rgb.save("watermarked_masked.png")
    print("加水印图像已保存：watermarked_masked.png")

    # 提取（可选同样的 mask 只查看嵌入区域）
    extracted = extract_watermark_repeat_mask(
        cover_img=cover,
        watermarked_img=result.watermarked_rgb,
        alpha=alpha,
        repeat_h=6,
        repeat_v=4,
        mask_img=mask,
        mask_threshold=None,
        mask_invert=False,
        return_masked_only=True if mask else False
    )
    extracted.save("extracted_masked.png")
    print("提取的水印已保存：extracted_masked.png")