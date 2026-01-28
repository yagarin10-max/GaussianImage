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
    # 例: "kl", "tgt0.4", "init1.0" を持つ手法をすべて表示
    # これにより、通常版、[No Clamp]版、[EMA]版などがすべてヒットし、自動で色分け・線種分けされます。
    ["kl", "tgt0.5", "init1.0"],
    ["kl", "tgt0.6", "init1.0"], 
    ["kl", "tgt0.8", "init1.0"], 
    ["kl", "tgt0.9", "init1.0"], 

]

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
            name = f"Mask ({reg}, tgt{tgt}, λ{lam}, init{init_val})"
            if "ada" in prefix:
                name += " [Ada]"
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
    figsize = (12, 6) if LEGEND_MODE == "outside" else (10, 6)
    plt.figure(figsize=figsize)
    
    methods = sorted(data.keys())
    plotted_count = 0
    
    # ベース名ごとの色管理用辞書
    base_name_color_map = {}
    
    # Matplotlibのデフォルト色サイクルを取得
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_idx = 0
    
    # マーカーリスト
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

        # --- 色とスタイルの決定ロジック ---
        
        # 1. ベース名を取得 (タグを除去して、純粋な手法名を抽出)
        # これにより "Mask (...) [No Clamp]" も "Mask (...)" として扱われる
        base_name = method.split(" [")[0]
        
        # 2. ベース名に対して色を割り当て (初めて出たベース名なら新色を付与)
        if base_name not in base_name_color_map:
            base_name_color_map[base_name] = colors[color_idx % len(colors)]
            color_idx += 1
        
        color = base_name_color_map[base_name]
        
        # 3. マーカーもベース名ごとに統一する (派生版と比較しやすくするため)
        # ベース名の登場順インデックスを使ってマーカーを決定
        base_idx = list(base_name_color_map.keys()).index(base_name)
        marker = markers[base_idx % len(markers)]
        
        # 4. 線種 (linestyle) でバリエーションを区別
        linestyle = '-' # デフォルト: 実線
        alpha = 0.9
        
        if "[No Clamp]" in method:
            linestyle = '--' # 点線
            alpha = 0.7      # 少し薄く
            
        if "[EMA]" in method:
            linestyle = '-.' # 一点鎖線
            alpha = 0.7
            
        if "[Score]" in method:
             linestyle = ':' # 点線(細)
             alpha = 0.7

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
            
            if SHOW_ERROR_BARS:
                plt.errorbar(
                    xs, ys, 
                    xerr=xerrs if x_axis_key == 'final' else None,
                    yerr=yerrs, 
                    marker=marker, 
                    linestyle=linestyle, 
                    color=color,    # 色を指定
                    linewidth=1.5,
                    capsize=3,
                    elinewidth=1.0,
                    alpha=alpha,
                    label=method
                )
            else:
                plt.plot(xs, ys, marker=marker, linestyle=linestyle, color=color, linewidth=2, label=method)
            
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
        plt.subplots_adjust(right=0.75) 
    elif LEGEND_MODE == "inside":
        plt.legend(loc='best', fontsize='small')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
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