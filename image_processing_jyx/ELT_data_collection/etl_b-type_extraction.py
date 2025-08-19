import os
import csv
import struct
import numpy as np
from PIL import Image


etl8info_file = "etl8info_labels.csv"
etl8_files = ["ETL8B2C1", "ETL8B2C2", "ETL8B2C3"]
output_dir = "output_etl8b2"
combined_csv = "ETL8B2_index.csv"

# --- mapping ---
def read_etl8info(file_path):
    mapping = {}
    if not os.path.exists(file_path):
        print(f"Cannot find mapping file: {file_path}")
        return mapping

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                filename = row[0].strip()
                label = row[1].strip()
                mapping[filename] = label
    print(f"Loaded {len(mapping)} mappings from {os.path.basename(file_path)}.")
    return mapping

# --- read one record (compat) ---
def read_record_ETL8B2(file, width=64, height=63):
    record_size = 512  # image data is 504 bytes
    s = file.read(record_size)
    if not s or len(s) < record_size:
        return None, None

    # JIS code (2 bytes) as integer
    jis_code = struct.unpack(">H", s[2:4])[0]

    # image: 64x63 1bpp => 4032 bits / 8 = 504 bytes
    img_data = np.unpackbits(np.frombuffer(s[8:8+504], dtype=np.uint8))
    img_data = img_data.reshape(height, width) * 255  # 0/1 to 0/255
    return jis_code, img_data

# --- JIS X 0208 (2 bytes) to Unicode via EUC-JP ---
def _decode_jis0208_to_unicode(jis_hi, jis_lo):
    try:
        return bytes([jis_hi + 0x80, jis_lo + 0x80]).decode("euc_jp")
    except Exception:
        return None

# --- read header + image ---
def _read_record_ETL8B2_full(file, width=64, height=63):
    """
    Return (header_dict, img_data) or (None, None).
    header_dict = {
        'writer_id': int,           # Serial Sheet Number (bytes 0-1, big-endian)
        'jis_code': str,            # '0xHHHH'
        'unicode': str|None,        # Unicode char
        'ascii_reading': str        # 4-byte ASCII
    }
    """
    record_size = 512
    s = file.read(record_size)
    if not s or len(s) < record_size:
        return None, None

    writer_id = struct.unpack(">H", s[0:2])[0]
    jis_hi, jis_lo = s[2], s[3]
    jis_code = f"0x{jis_hi:02X}{jis_lo:02X}"
    ascii_reading = s[4:8].decode("ascii", errors="replace")
    unicode_char = _decode_jis0208_to_unicode(jis_hi, jis_lo)

    img_bits = np.unpackbits(np.frombuffer(s[8:8+504], dtype=np.uint8))
    img_data = img_bits.reshape(height, width) * 255

    header = {
        "writer_id": writer_id,
        "jis_code": jis_code,
        "unicode": unicode_char,
        "ascii_reading": ascii_reading
    }
    return header, img_data

# --- process one file: write images into per-source subfolder to avoid name collisions ---
def process_ETL8B2_file(file_path, output_dir, mapping):
    """
    Image naming convention: {char_label}_{count}.png  (unchanged)
    Images are saved under: output_dir/<source_file>/...
    Returns rows for combined CSV:
      {
        "source_file": str,
        "image_filename": str,
        "writer_id": int,
        "jis_code": str,
        "unicode": str|None,
        "ascii_reading": str
      }
    """
    count = 0
    filename = os.path.basename(file_path)
    subdir = os.path.join(output_dir, filename)  # per-source subfolder
    os.makedirs(subdir, exist_ok=True)

    rows = []
    with open(file_path, "rb") as f:
        # skip dummy record (record 0)
        _ = f.read(512)

        while True:
            header, img_data = _read_record_ETL8B2_full(f)
            if header is None:
                break

            # use mapping[filename] or fallback UNK_{jis_code_int}
            if filename in mapping:
                char_label = mapping[filename]
            else:
                try:
                    jis_int = int(header["jis_code"], 16)
                except Exception:
                    jis_int = 0
                char_label = f"UNK_{jis_int}"

            image_filename = f"{char_label}_{count}.png"
            image_path = os.path.join(subdir, image_filename)

            # save image
            Image.fromarray(img_data.astype(np.uint8), mode="L").save(image_path)

            # append row (keep schema and names)
            rows.append({
                "source_file": filename,
                "image_filename": image_filename,
                "writer_id": header["writer_id"],
                "jis_code": header["jis_code"],
                "unicode": header["unicode"],
                "ascii_reading": header["ascii_reading"]
            })

            count += 1

    print(f"Finished {filename}: saved {count} images to {subdir}")
    return rows

# --- main: process C1/C2/C3 and write combined CSV ---
def main():
    mapping = read_etl8info(etl8info_file)
    os.makedirs(output_dir, exist_ok=True)

    combined_rows = []
    for etl_file in etl8_files:
        if not os.path.exists(etl_file):
            print(f"Skipped (not found): {etl_file}")
            continue
        rows = process_ETL8B2_file(etl_file, output_dir, mapping)
        combined_rows.extend(rows)

    combined_csv_path = os.path.join(output_dir, combined_csv)
    with open(combined_csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=["source_file", "image_filename", "writer_id", "jis_code", "unicode", "ascii_reading"]
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"Combined CSV exported to: {combined_csv_path}")
    print(f"Total images saved: {len(combined_rows)}")

if __name__ == "__main__":
    main()
