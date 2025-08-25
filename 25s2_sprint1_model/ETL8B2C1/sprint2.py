import csv

def read_etl8info(filename, output_csv):
    mapping = []
    with open(filename, "r", encoding="shift_jis") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                jis_code = parts[0]
                char = parts[1]
                mapping.append((jis_code, char))

    # 保存到 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["JIS_code", "character"])
        writer.writerows(mapping)

    print(f"✅ Saved mapping to {output_csv}")

if __name__ == "__main__":
    read_etl8info("ETL8INFO", "etl8info_labels.csv")
