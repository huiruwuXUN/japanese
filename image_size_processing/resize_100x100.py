from PIL import Image, ImageOps
from pathlib import Path

# 设置目录
root_dir = Path(r"C:\Users\Lenovo\Downloads\segmented_chars_img")
output_dir = Path(r"C:\Users\Lenovo\Downloads\resized_to_100x100")
output_dir.mkdir(parents=True, exist_ok=True)

count = 0

# 遍历所有 PNG 图像
for img_path in root_dir.rglob("*.png"):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # 确保没有透明通道

            # 获取原图尺寸
            w, h = img.size

            # 创建 100x100 白色背景图
            new_img = Image.new("RGB", (100, 100), color=(255, 255, 255))

            # 计算粘贴位置（使原图居中）
            left = (100 - w) // 2
            top = (100 - h) // 2

            # 粘贴到白底图上
            new_img.paste(img, (left, top))

            # 构造输出路径，保留原始子文件夹结构
            relative_path = img_path.relative_to(root_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存
            new_img.save(output_path)
            count += 1

    except Exception as e:
        print(f"⚠️ 无法处理 {img_path}: {e}")

print(f"\n✅ 已处理并保存 {count} 张图像到：{output_dir}")
