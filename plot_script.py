import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================================
# 設定 (Configuration)
# ==========================================
BASE_DIR = "checkpoints/kodak/"
TARGET_ITERATION = "50000"

OUTPUT_PSNR = "plots/comparison_psnr.png"
OUTPUT_SSIM = "plots/comparison_ssim.png"
OUTPUT_TABLE = "plots/summary_table.txt"

SHOW_ERROR_BARS = True
MAX_PLOT_POINTS = 40000
LEGEND_MODE = "outside"

# --- フィルタリング設定のポイント ---
# 「noclp」や「ema」をキーワードに入れると、それらが付いていない「ベースの手法」が除外されてしまいます。
# ベースと派生系（noclp, ema）を比較したい場合は、それらに「共通するキーワード」だけを指定してください。
FILTER_SPECS = [
    ["Baseline"], 
    # 2. KL: tgt0.4, 0.5, 0.6 をそれぞれ比較したい場合
    # (lam0.05, init1.0 は共通なので指定してもしなくても良いですが、絞るなら指定)
    # ["lam0.05", "init1.0"],
    # ["kl", "tgt0.5", "lam0.05", "init1.0"],
    ["kl", "tgt0.8", "lam0.05", "init1.0"],
    ["kl", "tgt0.9", "lam0.05", "init1.0"],
    
    # 3. L1 & L1sq: tgtは指定せず、reg, lam, initだけで指定する
    # これで tgt0.7(通常) と tgt0.6(noclp) の両方がヒットします
    # ["l1",   "lam0.05", "init1.0"],
    # ["l1sq", "lam0.05", "init1.0"]

]

EXCLUDE_KEYWORDS = ["No Clamp", "l1sq", "score"]  # 除外したいキーワードのリスト
# ==========================================
# データ処理関数
# ==========================================
def format_method_name(prefix, suffix):
    """
    prefix: _50000_xxxxx より前の部分
    suffix: _50000_xxxxx より後の部分
    """
    name = ""
    
    # --- 1. ベース名の決定 ---
    if "GaussianImage_Cholesky" in prefix and "wMask" not in prefix:
        name = "Baseline"
    elif "maskGI" in prefix:
        match = re.search(r"maskGI_Ch_(?:ada_)?([^_]+)_tgt([^_]+)_lam([^_]+)_init([^_]+)", prefix)
        if match:
            reg, tgt, lam, init_val = match.group(1), match.group(2), match.group(3), match.group(4)
            if "ada" in prefix:
                display_reg = f"ada_{reg}"
            else:
                display_reg = reg
            if reg in ["l1", "l1sq"]:
                name = f"Mask ({display_reg}, lam{lam}, init{init_val})"
            else:
                name = f"Mask ({display_reg}, tgt{tgt}, lam{lam}, init{init_val})"
        else:
            name = prefix
    else:
        name = prefix

    # --- 2. サフィックス（タグ）の付与 ---
    # 名称を [No Clip] -> [No Clamp] に変更
    if "noclp" in suffix or "noclp" in prefix:
        name += " [No Clamp]"
    
    if "ema" in suffix:
        name += " [EMA]"
        
    if "score" in suffix:
        name += " [Score]"

    return name

def parse_npy_logs(base_dir, target_iter):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # ディレクトリ検索
    search_pattern = os.path.join(base_dir, f"*_{target_iter}_*")
    exp_dirs = glob.glob(search_pattern)
    print(f"Found {len(exp_dirs)} experiment directories matching iter {target_iter}.")

    for exp_dir in exp_dirs:
        dir_name = os.path.basename(exp_dir)
        
        # 正規表現: 末尾に任意のサフィックス(group 3)を許容
        match = re.search(rf"^(.*)_{target_iter}_(\d+)(.*)$", dir_name)
        if not match: continue
            
        prefix = match.group(1)
        init_points = int(match.group(2))
        suffix = match.group(3)
        
        display_name = format_method_name(prefix, suffix)
        
        image_dirs = glob.glob(os.path.join(exp_dir, "kodim*"))
        
        for img_dir in image_dirs:
            npy_path = os.path.join(img_dir, "training.npy")
            if os.path.exists(npy_path):
                try:
                    npy_data = np.load(npy_path, allow_pickle=True).item()
                    psnr = npy_data.get('psnr', None)
                    ssim = npy_data.get('ms-ssim', None)
                    final_pts = npy_data.get('final_points', init_points)

                    if psnr is not None: data[display_name][init_points]['psnr'].append(psnr)
                    if ssim is not None: data[display_name][init_points]['ssim'].append(ssim)
                    data[display_name][init_points]['final_points'].append(final_pts)     
                except Exception as e:
                    print(f"Error reading {npy_path}: {e}")
    return data

def export_summary_table(data, output_file):
    lines = []
    header = f"{'Method':<60} | {'Init':<8} | {'Final(Avg)':<10} | {'Params(K)':<9} | {'PSNR(Avg)':<10} | {'PSNR(Std)':<10}"
    sep = "-" * len(header)
    lines.append(sep); lines.append(header); lines.append(sep)
    
    methods = sorted(data.keys())
    for method in methods:
        init_pts_list = sorted(data[method].keys())
        for init_pt in init_pts_list:
            stats = data[method][init_pt]
            psnr_avg = np.mean(stats['psnr']) if stats['psnr'] else 0
            psnr_std = np.std(stats['psnr']) if stats['psnr'] else 0
            final_pts_avg = np.mean(stats['final_points']) if stats['final_points'] else 0
            params_k = (final_pts_avg * 8) / 1000.0
            
            rows = f"{method:<60} | {init_pt:<8} | {final_pts_avg:<10.1f} | {params_k:<9.2f} | {psnr_avg:<10.4f} | {psnr_std:<10.4f}"
            lines.append(rows)
    lines.append(sep)
    output_text = "\n".join(lines)
    print("\n" + output_text + "\n")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(output_text)
    print(f"Summary table saved to: {output_file}")

# ==========================================
# プロット関数 (色リンク・スタイル制御)
# ==========================================
def plot_comparison(data, metric_key, y_label, title, output_file, x_axis_key='initial'):
    figsize = (15, 7) if LEGEND_MODE == "outside" else (10, 6)
    plt.figure(figsize=figsize)
    
    methods = sorted(data.keys())
    plotted_count = 0
    
    base_name_color_map = {}
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_idx = 0
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, method in enumerate(methods):
        # --- フィルタリング ---
        if FILTER_SPECS:
            is_match = False
            for group in FILTER_SPECS:
                method_lower = method.lower()
                # AND条件: グループ内のすべてのキーワードが含まれているか
                if all(kw.lower() in method_lower for kw in group):
                    is_match = True
                    break
            if not is_match:
                continue
        if EXCLUDE_KEYWORDS:
            is_excluded = False
            for kw in EXCLUDE_KEYWORDS:
                if kw.lower() in method.lower():
                    is_excluded = True
                    break
            if is_excluded: continue

        # --- 色とスタイルの決定ロジック ---
        base_name = method.split(" [")[0]
        if base_name not in base_name_color_map:
            base_name_color_map[base_name] = colors[color_idx % len(colors)]
            color_idx += 1
        
        color = base_name_color_map[base_name]
        
        linestyle = 'None' # デフォルトは線なし
        if "Baseline" in method:
            linestyle = '-' 
            if "[No Clamp]" in method: linestyle = '--'

        # 3. マーカー形状 (Shape): アルゴリズムの違いを表現
        # デフォルト: 丸 (Circle)
        marker = 'o' 
        
        has_ema = "[EMA]" in method
        has_score = "[Score]" in method
        
        if has_ema and has_score:
            marker = '*' # EMA + Score -> 星
        elif has_ema:
            marker = 'D' # EMAのみ -> ダイヤ
        elif has_score:
            marker = '^' # Scoreのみ -> 三角
        # なし -> 丸(o)

        # 4. 塗りつぶし (Fill): Clampの違いを表現
        # デフォルト: 塗りつぶし (Clampあり)
        marker_facecolor = color 
        
        if "[No Clamp]" in method:
            marker_facecolor = 'none' # 白抜き (No Clamp)
        # --- データ抽出 ---
        init_points_keys = sorted(data[method].keys())
        x_means, y_means, x_stds, y_stds = [], [], [], []
        
        for init_pt in init_points_keys:
            y_vals = data[method][init_pt][metric_key]
            if not y_vals: continue
            
            y_mean = np.mean(y_vals)
            y_std = np.std(y_vals)
            
            if x_axis_key == 'final':
                final_pts_vals = data[method][init_pt]['final_points']
                x_mean = np.mean(final_pts_vals)
                x_std = np.std(final_pts_vals)
            else:
                x_mean = init_pt
                x_std = 0
            
            if MAX_PLOT_POINTS is not None and x_mean > MAX_PLOT_POINTS:
                continue

            x_means.append(x_mean); y_means.append(y_mean)
            x_stds.append(x_std); y_stds.append(y_std)
        
        # --- プロット実行 ---
        if x_means:
            combined = sorted(zip(x_means, y_means, x_stds, y_stds), key=lambda x: x[0])
            xs, ys, xerrs, yerrs = zip(*combined)
            ms = 60 if marker == '*' else 50
            if SHOW_ERROR_BARS:
                plt.errorbar(
                    xs, ys, 
                    xerr=xerrs if x_axis_key == 'final' else None,
                    yerr=yerrs, 
                    fmt='none', 
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=3,
                    alpha=0.4,
                )
                if linestyle != 'None':
                    plt.plot(xs, ys, linestyle=linestyle, color=color, linewidth=1.5, alpha=0.6)
                
                plt.scatter(
                    xs, ys,
                    marker=marker,
                    s=ms,
                    facecolor=marker_facecolor,
                    edgecolor=color,
                    linewidths=1.2,
                    alpha=1.0,
                    zorder=10,
                    label=method
                )
            else:
                if linestyle != 'None':
                    plt.plot(xs, ys, linestyle=linestyle, color=color, linewidth=1.5, alpha=0.6)
                
                plt.scatter(
                    xs, ys, marker=marker, s=ms, 
                    facecolor=marker_facecolor, edgecolor=color, 
                    linewidth=1.2, label=method
                )
            plotted_count += 1

    if plotted_count == 0:
        print(f"Warning: No data matched filters. Skipping plot.")
        plt.close()
        return

    plt.xlabel("Average Final Gaussians" if x_axis_key == 'final' else "Initial Gaussians")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)

    if LEGEND_MODE == "outside":
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
        plt.subplots_adjust(right=0.6) 
    elif LEGEND_MODE == "inside":
        plt.legend(loc='best', fontsize='small')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()

def main():
    print("Parsing NPY files with advanced grouping...")
    data = parse_npy_logs(BASE_DIR, TARGET_ITERATION)
    
    if not data:
        print("No valid data found.")
        return

    export_summary_table(data, OUTPUT_TABLE)

    plot_comparison(data, 'psnr', 'Average PSNR (dB)', 
                   f"PSNR vs Final Points (Iter: {TARGET_ITERATION})", 
                   OUTPUT_PSNR, x_axis_key='final')

    plot_comparison(data, 'ssim', 'Average MS-SSIM', 
                   f"MS-SSIM vs Final Points (Iter: {TARGET_ITERATION})", 
                   OUTPUT_SSIM, x_axis_key='final')

if __name__ == "__main__":
    main()