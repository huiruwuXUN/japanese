import os
import glob
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
import argparse

# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class Metrics:
    path: str
    corners: int
    black_ratio: float
    label: str = "UNKNOWN"

# -----------------------------
# I/O 与预处理
# -----------------------------
def load_gray(path: str):
    """
    读取图像并转为灰度；支持包含非 ASCII 字符的路径。
    返回灰度图（uint8, HxW）或 None（读取失败）。
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data is None or data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def binarize_otsu(gray: np.ndarray, invert: bool):
    """
    Otsu 自动阈值 + 轻微高斯去噪 -> 二值图
    约定：默认“黑字白底”，黑色像素=0。若 invert=True 则取反适配“白字黑底”。
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        bw = 255 - bw
    return bw

def compute_black_ratio(bw: np.ndarray) -> float:
    """黑色像素占比 = (像素值==0) / 全部像素"""
    total = bw.size
    if total == 0:
        return 0.0
    black = np.count_nonzero(bw == 0)
    return black / total

# -----------------------------
# 角点检测
# -----------------------------
def count_corners(
    gray: np.ndarray,
    max_corners: int = 2000,
    quality: float = 0.01,
    min_distance: int = 3,
    block_size: int = 3,
    harris: bool = False
) -> int:
    """
    Shi–Tomasi 角点计数（可切换 Harris）：
      - max_corners：返回角点上限
      - quality：阈值系数，越小越敏感（更多角点）
      - min_distance：角点最小间距（像素）
      - block_size：角点评分邻域大小（奇数）
      - harris：True 则用 Harris 响应
    """
    corners = cv2.goodFeaturesToTrack(
        image=gray,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=harris
    )
    return 0 if corners is None else corners.shape[0]

# -----------------------------
# 分类与扫描
# -----------------------------
def classify_label(corners: int, black_ratio: float,
                   corner_th: int, black_th: float) -> str:
    """
    规则：角点 < corner_th 且 黑占比 < black_th => SIMPLE；否则 COMPLEX
    """
    return "SIMPLE" if (corners < corner_th and black_ratio < black_th) else "COMPLEX"

def scan_images(input_dir: str):
    """
    递归扫描所有常见图片扩展名。
    """
    exts = ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.bmp", "**/*.tif", "**/*.tiff", "**/*.webp"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(input_dir, e), recursive=True))
    paths = [p for p in paths if os.path.isfile(p) and not os.path.basename(p).startswith(".")]
    return sorted(paths)

def parse_hierarchy(path: str, input_dir: str):
    """
    从路径中解析 rc_id / page / cell（文件名不含扩展名）
    例：resized_to_100x100/RC04844/page_1/c1.png -> RC04844, page_1, c1
    """
    rel = os.path.relpath(path, input_dir)
    parts = rel.replace("\\", "/").split("/")
    rc_id = parts[0] if len(parts) >= 1 else ""
    page  = parts[1] if len(parts) >= 2 else ""
    stem  = os.path.splitext(parts[-1])[0] if parts else ""
    return rc_id, page, stem

def auto_thresholds(metrics_list, corner_q=0.4, black_q=0.4):
    """
    基于分位数的自适应阈值：
      corner_th = 角点数的 corner_q 分位
      black_th  = 黑占比的 black_q 分位
    """
    if not metrics_list:
        return 0, 0.0
    corners_arr = np.array([m.corners for m in metrics_list], dtype=float)
    black_arr   = np.array([m.black_ratio for m in metrics_list], dtype=float)
    corner_th = float(np.quantile(corners_arr, corner_q))
    black_th  = float(np.quantile(black_arr, black_q))
    return int(round(corner_th)), float(black_th)

# -----------------------------
# 可视化（可选）
# -----------------------------
def draw_corners_on_image(gray: np.ndarray, corners_xy: np.ndarray):
    """
    根据角点坐标在原灰度图上做可视化（画小圆点）。
    返回 BGR 图像。
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if corners_xy is not None and len(corners_xy) > 0:
        for x, y in corners_xy.reshape(-1, 2):
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    return vis

def get_corners_xy(
    gray: np.ndarray,
    max_corners: int,
    quality: float,
    min_distance: int,
    block_size: int,
    harris: bool
):
    pts = cv2.goodFeaturesToTrack(
        image=gray,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=harris
    )
    return None if pts is None else pts

# -----------------------------
# 主流程
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="用角点数量 + 黑色像素占比对图片进行简单/复杂分类（支持自动阈值与可视化）"
    )
    parser.add_argument("--input", required=True, help="输入图片根目录（将递归扫描）")

    # 阈值与策略
    parser.add_argument("--auto", action="store_true", help="启用自动阈值（基于分位数）")
    parser.add_argument("--corner-q", type=float, default=0.4, help="自动阈值：角点分位数（0~1）")
    parser.add_argument("--black-q", type=float, default=0.4, help="自动阈值：黑占比分位数（0~1）")
    parser.add_argument("--corner-th", type=int, default=150, help="固定阈值：角点阈值")
    parser.add_argument("--black-th", type=float, default=0.25, help="固定阈值：黑占比阈值")

    # 二值化方向
    parser.add_argument("--invert", action="store_true",
                        help="取反二值图（适配白字黑底）")

    # 角点检测参数
    parser.add_argument("--max-corners", type=int, default=2000)
    parser.add_argument("--quality", type=float, default=0.01)
    parser.add_argument("--min-distance", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--harris", action="store_true", help="使用 Harris 角点响应")

    # 可视化
    parser.add_argument("--save-viz", action="store_true", help="保存角点可视化图")
    parser.add_argument("--viz-dir", type=str, default="out_viz", help="可视化输出目录")

    args = parser.parse_args()

    # 扫描图片
    paths = scan_images(args.input)
    if not paths:
        print(f"[!] 在 {args.input} 下未找到图片。支持扩展名：png/jpg/jpeg/bmp/tif/tiff/webp")
        return

    # 先计算指标（用于自动阈值）
    metrics: list[Metrics] = []
    # 若需要可视化，准备角点坐标缓存（避免重复计算）
    need_xy = args.save_viz
    corners_xy_cache = {}

    for p in paths:
        gray = load_gray(p)
        if gray is None:
            print(f"[跳过] 无法读取：{p}")
            continue

        bw = binarize_otsu(gray, invert=args.invert)
        black_ratio = compute_black_ratio(bw)

        # 角点数（仅计数）
        corners = count_corners(
            gray,
            max_corners=args.max_corners,
            quality=args.quality,
            min_distance=args.min_distance,
            block_size=args.block_size,
            harris=args.harris
        )

        metrics.append(Metrics(path=p, corners=corners, black_ratio=black_ratio))

        # 如需可视化，获取角点坐标
        if need_xy:
            pts = get_corners_xy(
                gray,
                max_corners=args.max_corners,
                quality=args.quality,
                min_distance=args.min_distance,
                block_size=args.block_size,
                harris=args.harris
            )
            corners_xy_cache[p] = (gray, pts)

    # 阈值
    if args.auto:
        corner_th, black_th = auto_thresholds(metrics, args.corner_q, args.black_q)
        print(f"[自动阈值] corner_th={corner_th}, black_th={black_th:.4f} "
              f"(corner_q={args.corner_q}, black_q={args.black_q})")
    else:
        corner_th, black_th = args.corner_th, args.black_th
        print(f"[固定阈值] corner_th={corner_th}, black_th={black_th:.4f}")

    # 分类并导出 CSV
    rows = []
    for m in metrics:
        m.label = classify_label(m.corners, m.black_ratio, corner_th, black_th)
        rc_id, page, cell = parse_hierarchy(m.path, args.input)
        rows.append({
            "rc_id": rc_id,
            "page": page,
            "cell": cell,
            "relpath": os.path.relpath(m.path, args.input).replace("\\", "/"),
            "corners": m.corners,
            "black_ratio": round(m.black_ratio, 6),
            "label": m.label
        })

    df = pd.DataFrame(rows)
    out_csv = "results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = (df.groupby(["rc_id", "page"])
                 .agg(n=("label", "size"),
                      simple=("label", lambda s: int((s == "SIMPLE").sum())),
                      complex=("label", lambda s: int((s == "COMPLEX").sum())),
                      avg_corners=("corners", "mean"),
                      avg_black=("black_ratio", "mean"))
                 .reset_index())
    summary_csv = "summary_by_rc_page.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"[完成] 共处理 {len(df)} 张图片。")
    print(f"明细: {out_csv}")
    print(f"汇总: {summary_csv}")

        # 可视化保存
    if args.save_viz:
        # 先把明细表变成一个 {relpath: row_dict} 的索引，便于查值
        rows_by_rel = {
            r["relpath"]: r
            for r in df.to_dict(orient="records")
        }

        for p, (gray, pts) in corners_xy_cache.items():
            vis = draw_corners_on_image(gray, pts)

            # 先把相对路径准备好（避免在 f-string 里写带反斜杠的 replace）
            relpath = os.path.relpath(p, args.input).replace("\\", "/")

            # 从索引里拿该图的指标
            row = rows_by_rel.get(relpath, None)
            if row is None:
                # 理论上不会发生；保险起见
                row = {"corners": 0, "black_ratio": 0.0, "label": "UNKNOWN"}

            corners_val = int(row["corners"])
            black_val   = float(row["black_ratio"])
            label_val   = str(row["label"])

            info_text = f"corners={corners_val}  black={black_val:.3f}  {label_val}"
            cv2.putText(
                vis, info_text,
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 0, 0), 1, cv2.LINE_AA
            )

            # 构造输出路径（保持相对层级）
            out_path = os.path.join(args.viz_dir, relpath)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # 统一用原扩展名保存（若不在 png/jpg/jpeg 列表则回退 png）
            ext = os.path.splitext(out_path)[1].lower()
            if ext not in [".png", ".jpg", ".jpeg"]:
                ext = ".png"
                out_path = os.path.splitext(out_path)[0] + ext

            ok, buf = cv2.imencode(ext, vis)
            if ok:
                with open(out_path, "wb") as f:
                    f.write(buf.tobytes())

        print(f"[可视化] 已保存到目录：{args.viz_dir}")


if __name__ == "__main__":
    import sys, os

    if len(sys.argv) == 1:
        # 没有传入命令行参数时，自动使用一组“调试用默认参数”
        # 按需修改 "--input" 的路径（相对/绝对都可）
        default_args = [
    "--input", "C:/Users/Lenovo/Downloads/resized_to_100x100",
    "--auto",
    "--corner-q", "0.4",#百分比占比
    "--black-q", "0.4",
    "--save-viz",
    "--viz-dir", "out_viz",
]
        print("[info] No CLI args detected; using defaults:\n   ",
              " ".join(default_args))
        sys.argv.extend(default_args)

    # （可选）友好提示：检查输入目录是否存在
    try:
        idx = sys.argv.index("--input")
        in_dir = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else ""
        if in_dir and not os.path.isdir(in_dir):
            print(f"[warn] --input 路径不存在：{in_dir}  （请确认工作目录或改为绝对路径）")
    except ValueError:
        pass

    main()

