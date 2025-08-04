from PIL import Image
from pathlib import Path

# 设置根目录
root_dir = Path(r"C:\Users\Lenovo\Downloads\segmented_chars_img")

max_pixels = 0
max_image_path = None
max_size = (0, 0)

# 遍历所有 PNG 图片
for img_path in root_dir.rglob("*.png"):
    try:
        with Image.open(img_path) as img:
            width, height = img.size

            # ❌ 忽略宽度或高度为 200 的图片
            if width >= 100 or height >= 100:
                continue

            pixels = width * height
            if pixels > max_pixels:
                max_pixels = pixels
                max_image_path = img_path
                max_size = (width, height)
    except Exception as e:
        print(f"⚠️ 无法打开 {img_path}: {e}")

# 输出结果
if max_image_path:
    print("\n🎯 最大尺寸图片（忽略宽或高为 200 的图片）：")
    print(f"🖼️ 文件名：{max_image_path.name}")
    print(f"📐 尺寸：{max_size[0]} × {max_size[1]}")
    print(f"📦 像素数：{max_pixels}")
    print(f"📂 路径：{max_image_path.resolve()}")
else:
    print("❌ 没有找到符合条件的图片。")
