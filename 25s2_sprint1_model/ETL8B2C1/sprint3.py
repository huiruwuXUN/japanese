import os
import csv
import struct
import numpy as np
from PIL import Image

# --- 配置 ---
etl8info_file = "etl8info_labels.csv"  # 你的映射文件
etl8_files = ["ETL8B2C3"]              # 只跑 C1
output_dir = "output_etl8b2_C3"

# --- 读取映射 ---
def read_etl8info(file_path):
    mapping = {}
    if not os.path.exists(file_path):
        print(f"❌ Cannot find mapping file: {file_path}")
        return mapping

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:
            if len(row) >= 2:
                filename = row[0].strip()
                label = row[1].strip()
                mapping[filename] = label
    print(f"✅ Loaded {len(mapping)} mappings from {os.path.basename(file_path)}.")
    return mapping

# --- 读取一个记录 ---
def read_record_ETL8B2(file, width=64, height=63):
    # B-Type 每条记录大小固定
    record_size = 512  # 根据ETL8B2结构，图像数据部分是504字节，头部若干
    s = file.read(record_size)
    if not s or len(s) < record_size:
        return None, None

    # 解码 JIS 编码 (2 bytes)
    jis_code = struct.unpack(">H", s[2:4])[0]

    # 图像数据部分：从头部之后取出504字节（64x63位图 => 4032 bits / 8 = 504 bytes）
    img_data = np.unpackbits(np.frombuffer(s[8:8+504], dtype=np.uint8))
    img_data = img_data.reshape(height, width) * 255  # 0/1 转 0/255
    return jis_code, img_data

# --- 处理整个文件 ---
def process_ETL8B2_file(file_path, output_dir, mapping):
    count = 0
    filename = os.path.basename(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, "rb") as f:
        while True:
            rec = read_record_ETL8B2(f)
            if rec[0] is None:
                break
            jis_code, img_data = rec
            # 用 JIS code 映射到字符（这里你可以换成 mapping）
            char_label = mapping.get(filename, f"UNK_{jis_code}")
            img = Image.fromarray(img_data.astype(np.uint8), mode="L")
            img.save(os.path.join(output_dir, f"{char_label}_{count}.png"))
            count += 1

    print(f"✅ Finished {filename}, saved {count} images.")

# --- 主程序 ---
def main():
    mapping = read_etl8info(etl8info_file)
    for etl_file in etl8_files:
        process_ETL8B2_file(etl_file, output_dir, mapping)

if __name__ == "__main__":
    main()
