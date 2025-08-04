from PIL import Image
from pathlib import Path

# 设置根目录路径
root_dir = Path(r"C:\Users\Lenovo\Downloads\segmented_chars_img")

# 存储所有图片信息 (路径, 像素数, 尺寸)
image_info = []

# 遍历所有 PNG 文件（递归）
for img_path in root_dir.rglob("*.png"):
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            pixels = width * height
            image_info.append((img_path, pixels, (width, height)))
    except Exception as e:
        print(f"⚠️ 无法打开 {img_path}: {e}")

# 检查是否找到图片
if not image_info:
    print("❌ 没有找到 PNG 图片。")
    exit()

# 按像素数降序排列
image_info.sort(key=lambda x: x[1], reverse=True)

# 输出前10张最大图片
top_n = 10
print(f"\n📈 像素最大的前 {top_n} 张图片：\n")

for i, (path, pixels, size) in enumerate(image_info[:top_n], start=1):
    print(f"{i}. 🖼️ {path.name}")
    print(f"   📐 尺寸：{size[0]} × {size[1]}，像素数：{pixels}")
    print(f"   📂 路径：{path.resolve()}\n")
