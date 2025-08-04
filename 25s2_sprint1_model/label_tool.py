import csv
import cv2
from pathlib import Path

# 路径设置
base_dir = Path(r"C:\Users\Lenovo\Downloads\resized_to_100x100")
csv_path = Path("labels.csv")

# 读取 CSV
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# 标注循环
for i, row in enumerate(rows):
    filepath, label = row
    if label.strip() != "":
        continue

    full_path = base_dir / filepath
    if not full_path.exists():
        print(f"❌ 文件不存在：{full_path}")
        continue

    img = cv2.imread(str(full_path))
    if img is None:
        print(f"⚠️ 图像无法读取：{filepath}")
        continue

    # 放大显示 & 添加边框
    zoom = 4
    resized = cv2.resize(img, (img.shape[1]*zoom, img.shape[0]*zoom), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(resized, (0, 0), (resized.shape[1]-1, resized.shape[0]-1), (0, 255, 0), 2)

    cv2.imshow("Press 0 / 1 / s=skip / q=quit", resized)

    while True:
        key = cv2.waitKey(0)
        if key == ord('0'):
            rows[i][1] = '0'
            break
        elif key == ord('1'):
            rows[i][1] = '1'
            break
        elif key == ord('s'):
            print(f"⏭️ 跳过 {filepath}")
            break
        elif key == ord('q'):
            print("🛑 退出程序，保存标注...")
            break

    cv2.destroyAllWindows()

    if key == ord('q'):
        break

# 保存 CSV
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("✅ 所有标注已保存！")
