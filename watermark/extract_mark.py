import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pywt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# 导入必要的转换函数
def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32)

def rgb_to_ycbcr_np(rgb: np.ndarray) -> np.ndarray:
    transform = np.array([
        [ 0.299,     0.587,     0.114   ],
        [-0.168736, -0.331264,  0.5     ],
        [ 0.5,      -0.418688, -0.081312]
    ], dtype=np.float32)
    shift = np.array([0, 128, 128], dtype=np.float32)
    return rgb @ transform.T + shift

def extract_watermark_simple(cover_img_path: str, watermarked_img_path: str, alpha: float = 25.0):
    """
    从加水印的图像中提取水印
    参数:
    - cover_img_path: 原始图像路径
    - watermarked_img_path: 加水印图像路径
    - alpha: 嵌入时使用的强度参数
    返回:
    - extracted_watermark: 提取的水印图像 (PIL Image)
    """
    
    # 1. 加载图像
    cover = Image.open(cover_img_path).convert("RGB")
    watermarked = Image.open(watermarked_img_path).convert("RGB")
    
    print(f"原图尺寸: {cover.size}")
    print(f"加水印图尺寸: {watermarked.size}")
    
    # 2. 转换到YCbCr色彩空间，提取Y通道
    cover_rgb = pil_to_np(cover)
    cover_ycbcr = rgb_to_ycbcr_np(cover_rgb)
    Y_cover = cover_ycbcr[..., 0]
    
    watermarked_rgb = pil_to_np(watermarked)
    watermarked_ycbcr = rgb_to_ycbcr_np(watermarked_rgb)
    Y_watermarked = watermarked_ycbcr[..., 0]
    
    # 3. 对Y通道进行小波变换
    coeffs_cover = pywt.dwt2(Y_cover, 'haar')
    LL_cover, (LH_cover, HL_cover, HH_cover) = coeffs_cover
    
    coeffs_watermarked = pywt.dwt2(Y_watermarked, 'haar')
    LL_watermarked, (LH_wm, HL_wm, HH_wm) = coeffs_watermarked
    
    print(f"LL子带尺寸: {LL_cover.shape}")
    
    # 4. 提取水印 (从LL子带的差异中恢复)
    # 公式: watermark = (LL_watermarked - LL_cover) / alpha
    watermark_diff = (LL_watermarked - LL_cover) / (alpha + 1e-8)
    
    # 5. 将水印数据归一化到 [0, 255] 范围
    # 假设原始水印被归一化到 [-0.5, 0.5] 范围
    watermark_recovered = (watermark_diff + 0.5) * 255.0
    watermark_recovered = np.clip(watermark_recovered, 0, 255).astype(np.uint8)
    
    # 6. 转换为PIL图像
    extracted_watermark = Image.fromarray(watermark_recovered, mode='L')
    
    print(f"提取的水印尺寸: {extracted_watermark.size}")
    
    return extracted_watermark

def calculate_similarity(original_wm_path: str, extracted_wm: Image.Image):
    """计算原始水印和提取水印的相似度"""
    
    # 加载原始水印
    original_wm = Image.open(original_wm_path).convert('L')
    
    # 转换为numpy数组
    original_array = np.array(original_wm)
    extracted_array = np.array(extracted_wm)
    
    # 调整尺寸匹配
    if original_array.shape != extracted_array.shape:
        from skimage.transform import resize
        original_resized = resize(original_array, extracted_array.shape, preserve_range=True).astype(np.uint8)
    else:
        original_resized = original_array
    
    # 计算相关系数
    correlation = np.corrcoef(original_resized.flatten(), extracted_array.flatten())[0, 1]
    
    # 计算SSIM
    ssim_value = ssim(original_resized, extracted_array, data_range=255)
    
    # 计算PSNR
    mse = np.mean((original_resized.astype(float) - extracted_array.astype(float)) ** 2)
    psnr_value = 20 * np.log10(255.0 / np.sqrt(mse + 1e-8)) if mse > 0 else float('inf')
    
    return correlation, ssim_value, psnr_value

def create_comparison_plot(cover_path: str, watermarked_path: str, 
                          original_wm_path: str, extracted_wm: Image.Image):
    """创建四张图的对比展示"""
    
    # 设置中文字体（如果有的话）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Watermark Extraction Comparison', fontsize=16, fontweight='bold')
    
    # 加载图像
    cover = Image.open(cover_path)
    watermarked = Image.open(watermarked_path)
    original_wm = Image.open(original_wm_path).convert('L')
    
    # 1. 原始图像
    axes[0, 0].imshow(cover)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. 加水印图像
    axes[0, 1].imshow(watermarked)
    axes[0, 1].set_title('Watermarked Image', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # 3. 原始水印
    axes[1, 0].imshow(original_wm, cmap='gray')
    axes[1, 0].set_title('Original Watermark', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 4. 提取的水印
    axes[1, 1].imshow(extracted_wm, cmap='gray')
    axes[1, 1].set_title('Extracted Watermark', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = 'watermark_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"对比图已保存: {output_path}")
    
    # 显示图像（如果在支持的环境中）
    try:
        plt.show()
    except:
        print("无法显示图像（可能在无GUI环境中运行）")
    
    return output_path

def main():
    """主函数：演示完整的水印提取流程"""
    
    print("=== 水印提取演示 ===\n")
    
    # 文件路径（请根据实际情况修改）
    cover_path = "cover.jpg"
    watermarked_path = "watermarked_masked.png"  # 或者你的加水印图像文件名
    original_wm_path = "watermark.png"
    
    # 检查文件是否存在
    required_files = [cover_path, watermarked_path, original_wm_path]
    missing_files = []
    
    for file_path in required_files:
        try:
            Image.open(file_path)
        except FileNotFoundError:
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 错误：缺少以下文件 {missing_files}")
        print("请确保以下文件存在：")
        print(f"  - {cover_path} (原始图像)")
        print(f"  - {watermarked_path} (加水印的图像)")
        print(f"  - {original_wm_path} (原始水印)")
        return
    
    try:
        # 1. 提取水印（使用与嵌入时相同的alpha值）
        print("1. 正在提取水印...")
        alpha = 25.0  # 请使用与嵌入时相同的alpha值
        extracted_wm = extract_watermark_simple(cover_path, watermarked_path, alpha)
        
        # 保存提取的水印
        extracted_path = "extracted_watermark.png"
        extracted_wm.save(extracted_path)
        print(f"✅ 水印提取完成，已保存: {extracted_path}")
        
        # 2. 计算相似度
        print("\n2. 计算水印相似度...")
        correlation, ssim_val, psnr_val = calculate_similarity(original_wm_path, extracted_wm)
        
        print(f"相关系数: {correlation:.4f}")
        print(f"SSIM: {ssim_val:.4f}")
        print(f"PSNR: {psnr_val:.2f} dB")
        
        # 3. 评估提取质量
        print("\n3. 提取质量评估:")
        if correlation > 0.5:
            print("✅ 优秀 - 水印提取成功，可以清晰识别")
        elif correlation > 0.3:
            print("⚠️ 良好 - 水印部分可识别，基本可用")
        elif correlation > 0.1:
            print("⚠️ 一般 - 可以检测到水印信号，但识别困难")
        else:
            print("❌ 较差 - 水印信号微弱，难以识别")
        
        # 4. 创建对比图
        print("\n4. 创建对比展示...")
        comparison_path = create_comparison_plot(cover_path, watermarked_path, 
                                               original_wm_path, extracted_wm)
        
        print(f"\n=== 提取完成 ===")
        print(f"生成的文件:")
        print(f"  - {extracted_path} (提取的水印)")
        print(f"  - {comparison_path} (对比图)")
        
        # 5. 给出改进建议
        if correlation < 0.3:
            print(f"\n💡 改进建议:")
            print(f"  - 尝试增加alpha值 (当前: {alpha})")
            print(f"  - 检查是否使用了相同的嵌入参数")
            print(f"  - 确保原图和加水印图像匹配")
    
    except Exception as e:
        print(f"❌ 提取过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()