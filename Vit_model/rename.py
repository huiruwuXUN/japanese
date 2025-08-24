# check_filenames_only.py
# 仅检查：CSV (source_file + image_filename) 与目录结构是否一一对应
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd

# 允许的图片扩展名（小写）
ACCEPT_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

def list_images_flat(folder: Path) -> List[str]:
    """仅列出该文件夹下的一层图片文件名（不递归），按名字排序"""
    names = []
    for p in folder.iterdir():
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in ACCEPT_EXTS:
            names.append(p.name)
    return sorted(names, key=str.lower)

def main():
    ap = argparse.ArgumentParser(description="只检查 CSV 与多子文件夹文件名是否一一对应（不重命名）")
    ap.add_argument("--root", type=str, default="output_etl8b3", help="图片根目录（其下有 ETL8B2C1/ETL8B2C2/...）")
    ap.add_argument("--csv", type=str, default="ETL8B2_index.csv", help="CSV 路径，需含 source_file 和 image_filename 列")
    ap.add_argument("--report_dir", type=str, default="", help="可选：导出差异报告到该目录（missing/extra）")
    args = ap.parse_args()

    root = Path(args.root)
    csv_path = Path(args.csv)
    if not root.exists():
        print(f"[ERROR] 根目录不存在：{root}")
        sys.exit(1)
    if not csv_path.exists():
        print(f"[ERROR] CSV 不存在：{csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = {"source_file", "image_filename"}
    if not required.issubset(df.columns):
        print(f"[ERROR] CSV 缺少必要列 {required}；实际列：{list(df.columns)}")
        sys.exit(1)

    # 统计 CSV 内部重复
    dup_all = df["image_filename"].value_counts()
    dup_all = dup_all[dup_all > 1]
    if len(dup_all) > 0:
        print(f"[WARN] CSV 中发现 {len(dup_all)} 个重复文件名（跨 source_file 统计）示例：")
        print(dup_all.head(10))

    # 按 source_file 分组
    groups: Dict[str, List[str]] = {}
    for src, g in df.groupby("source_file"):
        groups[str(src)] = sorted(g["image_filename"].astype(str).tolist(), key=str.lower)

    # 准备可选报告目录
    if args.report_dir:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
    else:
        report_dir = None

    total_expected = 0
    total_actual = 0
    total_missing = 0
    total_extra = 0
    count_mismatch = 0
    name_mismatch_but_equal_count = 0

    print("============ 开始逐文件夹检查 ============")
    for sub in sorted(groups.keys()):
        expected = groups[sub]
        total_expected += len(expected)

        folder = root / sub
        print(f"\n[Folder] {sub}")
        if not folder.exists():
            print(f"  [ERROR] 子文件夹不存在：{folder}，其 CSV 期望 {len(expected)} 张。")
            total_missing += len(expected)  # 全部视为缺失
            count_mismatch += 1
            # 导出报告
            if report_dir:
                pd.DataFrame({"missing_in_dir": expected}).to_csv(report_dir / f"{sub}_missing.csv", index=False)
            continue

        actual = list_images_flat(folder)
        total_actual += len(actual)

        set_exp = set(expected)
        set_act = set(actual)
        missing = sorted(list(set_exp - set_act), key=str.lower)   # CSV 有但目录没有
        extra   = sorted(list(set_act - set_exp), key=str.lower)   # 目录有但 CSV 没有

        print(f"  CSV 期望: {len(expected)} | 目录实际: {len(actual)}")

        if len(expected) != len(actual):
            print("  [COUNT MISMATCH] 数量不一致。")
            count_mismatch += 1
        else:
            if missing or extra:
                print("  [NAME MISMATCH] 数量一致但文件名不一致。")
                name_mismatch_but_equal_count += 1
            else:
                print("  [OK] 完全一致。")

        if missing:
            total_missing += len(missing)
            print(f"  缺失 {len(missing)}：示例 {missing[:5]}")
            if report_dir:
                pd.DataFrame({"missing_in_dir": missing}).to_csv(report_dir / f"{sub}_missing.csv", index=False)
        if extra:
            total_extra += len(extra)
            print(f"  多余 {len(extra)}：示例 {extra[:5]}")
            if report_dir:
                pd.DataFrame({"extra_in_dir": extra}).to_csv(report_dir / f"{sub}_extra.csv", index=False)

        # 额外提示：目录中的非图片文件/隐藏文件会被忽略
        others = [p.name for p in folder.iterdir() if p.is_file() and (p.suffix.lower() not in ACCEPT_EXTS or p.name.startswith("."))]
        if others:
            print(f"  [NOTE] 发现 {len(others)} 个非图片/隐藏文件（已忽略），示例：{others[:3]}")

    print("\n============ 汇总 ============")
    print(f"CSV 期望总数：{total_expected}")
    print(f"目录实际总数：{total_actual}")
    print(f"总缺失数（CSV有但目录没有）：{total_missing}")
    print(f"总多余数（目录有但CSV没有）：{total_extra}")
    print(f"数量不一致的文件夹数：{count_mismatch}")
    print(f"数量一致但名字不一致的文件夹数：{name_mismatch_but_equal_count}")

    # 全局 also：检查 root 下是否存在 CSV 未提到的子文件夹
    known = set(groups.keys())
    existing_subs = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}
    extra_folders = sorted(list(existing_subs - known))
    if extra_folders:
        print(f"\n[NOTE] 目录中存在 CSV 未列出的子文件夹：{extra_folders}")

if __name__ == "__main__":
    main()
