from PIL import Image
from pathlib import Path

# 设置根目录
root_dir = Path(r"C:\Users\Lenovo\Downloads\segmented_chars_img")

delete_list = []

# 统计并收集所有要删除的图像路径
for img_path in root_dir.rglob("*.png"):
    try:
        with Image.open(img_path) as img:
            width, height = img.size

            # 🚫 如果宽度或高度大于等于100 → 标记为删除
            if width >= 100 or height >= 100:
                delete_list.append(img_path)
    except Exception as e:
        print(f"⚠️ 无法打开 {img_path}: {e}")

# 显示统计信息
print(f"\n🧾 将删除的图片数量：{len(delete_list)}")
if delete_list:
    confirm = input("⚠️ 确认删除这些文件？输入 yes 执行删除：")
    if confirm.lower() == "yes":
        for path in delete_list:
            try:
                path.unlink()
                print(f"✅ 已删除：{path}")
            except Exception as e:
                print(f"❌ 删除失败：{path}，错误：{e}")
    else:
        print("🚫 已取消删除操作。")
else:
    print("✅ 没有符合删除条件的图片。")
