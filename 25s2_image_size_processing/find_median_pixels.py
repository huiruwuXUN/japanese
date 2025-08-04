from PIL import Image
from pathlib import Path
import statistics

# 目录路径
root_dir = Path(r"C:\Users\Lenovo\Downloads\segmented_chars_img")

# 存储所有图片信息 (路径, 像素数, 尺寸)
image_info = []

# 遍历所有 PNG 图片
for img_path in root_dir.rglob("*.png"):
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            pixels = width * height
            image_info.append((img_path, pixels, (width, height)))
    except Exception as e:
        print(f"⚠️ 无法打开 {img_path}: {e}")

# 如果没有找到图片
if not image_info:
    print("❌ 没有找到 PNG 图片。")
    exit()

# 排序（按像素数从小到大）
image_info.sort(key=lambda x: x[1])

# 计算中位数
pixel_values = [info[1] for info in image_info]
median_pixels = statistics.median(pixel_values)

# 找出最接近中位数的图（差值最小）
closest_img = min(image_info, key=lambda x: abs(x[1] - median_pixels))

# 输出中位数结果
print(f"\n📊 中位数像素数：{int(median_pixels)} 像素")
print(f"🖼️ 最接近中位数的图片：{closest_img[0].name}")
print(f"📐 尺寸：{closest_img[2][0]} × {closest_img[2][1]}")
print(f"📂 路径：{closest_img[0].resolve()}")
