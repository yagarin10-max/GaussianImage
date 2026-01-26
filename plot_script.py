import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================================
# 設定
# ==========================================
BASE_DIR = "checkpoints/kodak/"
TARGET_ITERATION = "50000"  # 比較対象とするIteration

# 保存ファイル名
OUTPUT_PSNR = "plots/comparison_psnr.png"
OUTPUT_SSIM = "plots/comparison_ssim.png"
OUTPUT_FINAL_POINTS = "plots/comparison_final_points.png" # 追加: 最終点数の推移も見れるように
OUTPUT_TABLE = "plots/summary_table.txt"

SHOW_ERROR_BARS = True
MAX_PLOT_POINTS = 40000
LEGEND_MODE = "outside"
FILTER_SPECS = [
    ["Baseline"],          # グループ1: "Baseline" を含むならOK
    ["kl", "tgt0.4"],      # グループ2: "kl" と "tgt0.7" の両方を含むならOK
    # ["l1", "tgt0.7"]     # 必要ならグループ3を追加...
]

def format_method_name(raw_name):
    """
    ディレクトリ名をグラフの凡例用に短く整形する
    """
    # Baseline (Choleskyなど)
    if "GaussianImage_Cholesky" in raw_name and "wMask" not in raw_name:
        return "Baseline"
    
    # Mask手法 (maskGI...)
    # maskGI_Ch_kl_tgt0.7_lam0.005_init2.0 -> Mask (kl, tgt0.7, λ0.005)
    if "maskGI" in raw_name:
        match = re.search(r"maskGI_Ch_([^_]+)_tgt([^_]+)_lam([^_]+)_init([^_]+)", raw_name)
        if match:
            reg = match.group(1)
            tgt = match.group(2)
            lam = match.group(3)
            init = match.group(4) # 必要ならinitも表示
            return f"Mask ({reg}, tgt{tgt}, λ{lam}, init{init})"
            
    return raw_name

def parse_npy_logs(base_dir, target_iter):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    search_pattern = os.path.join(base_dir, f"*_{target_iter}_*")
    exp_dirs = glob.glob(search_pattern)
    
    print(f"Found {len(exp_dirs)} experiment directories matching iter {target_iter}.")

    for exp_dir in exp_dirs:
        dir_name = os.path.basename(exp_dir)
        match = re.search(rf"^(.*)_{target_iter}_(\d+)$", dir_name)
        if not match:
            continue
            
        raw_method_name = match.group(1)
        init_points = int(match.group(2))
        display_name = format_method_name(raw_method_name)
        
        image_dirs = glob.glob(os.path.join(exp_dir, "kodim*"))
        
        for img_dir in image_dirs:
            npy_path = os.path.join(img_dir, "training.npy")
            
            if os.path.exists(npy_path):
                try:
                    npy_data = np.load(npy_path, allow_pickle=True).item()
                    
                    # ---------------------------------------------------
                    # データの取得と保険処理（フォールバック）
                    # ---------------------------------------------------
                    psnr = npy_data.get('psnr', None)
                    ssim = npy_data.get('ms-ssim', None)
                    
                    # final_pointsの取得ロジック
                    if 'final_points' in npy_data:
                        # キーが存在する場合はそれを使う (Mask手法など)
                        final_pts = npy_data['final_points']
                    else:
                        # キーが存在しない場合は、初期点数をそのまま使う (Baselineなど)
                        final_pts = init_points 

                    # リストに追加
                    if psnr is not None:
                        data[display_name][init_points]['psnr'].append(psnr)
                    if ssim is not None:
                        data[display_name][init_points]['ssim'].append(ssim)
                    
                    # final_ptsは None になり得ないのでそのまま追加
                    data[display_name][init_points]['final_points'].append(final_pts)
                        
                except Exception as e:
                    print(f"Error reading {npy_path}: {e}")

    return data

# ==========================================
# 表作成・出力関数 (New!)
# ==========================================
def export_summary_table(data, output_file):
    lines = []
    
    # ヘッダー作成
    # Method | Init | Final(Avg) | Params | Params(K) | PSNR | SSIM
    header = f"{'Method':<35} | {'Init':<8} | {'Final(Avg)':<10} | {'Params':<10} | {'Params(K)':<9} | {'PSNR':<8} | {'SSIM':<8}"
    sep = "-" * len(header)
    
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    
    # データをリスト化してソート (手法名 -> 初期点数 の順)
    rows = []
    methods = sorted(data.keys())
    
    for method in methods:
        init_pts_list = sorted(data[method].keys())
        for init_pt in init_pts_list:
            stats = data[method][init_pt]
            
            # 平均計算
            psnr_avg = np.mean(stats['psnr']) if stats['psnr'] else 0
            ssim_avg = np.mean(stats['ssim']) if stats['ssim'] else 0
            final_pts_avg = np.mean(stats['final_points']) if stats['final_points'] else 0
            
            # Params計算 (Gaussian数 * 8)
            params = final_pts_avg * 8
            params_k = params / 1000.0
            
            rows.append({
                'method': method,
                'init': init_pt,
                'final': final_pts_avg,
                'params': params,
                'params_k': params_k,
                'psnr': psnr_avg,
                'ssim': ssim_avg
            })

    # 行を追加
    for r in rows:
        line = f"{r['method']:<35} | {r['init']:<8} | {r['final']:<10.1f} | {r['params']:<10.1f} | {r['params_k']:<9.2f} | {r['psnr']:<8.4f} | {r['ssim']:<8.6f}"
        lines.append(line)
    
    lines.append(sep)
    
    # 結果の結合
    output_text = "\n".join(lines)
    
    # 1. コンソールに出力
    print("\n" + "="*20 + " SUMMARY TABLE " + "="*20)
    print(output_text)
    print("="*55 + "\n")
    
    # 2. ファイルに保存
    with open(output_file, "w") as f:
        f.write(output_text)
    print(f"Summary table saved to: {output_file}")

# ==========================================
# プロット関数 (X軸を動的に計算するように修正)
# ==========================================
def plot_comparison(data, metric_key, y_label, title, output_file, x_axis_key='initial'):
    """
    x_axis_key: 'initial' なら初期点数(固定)をX軸に、
                'final' なら実際の最終点数(平均)をX軸に使用する
    """
    figsize = (12, 6) if LEGEND_MODE == "outside" else (10, 6)
    plt.figure(figsize=figsize)

    methods = sorted(data.keys())
    plotted_count = 0
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    if not methods:
        print(f"No data found for metric: {metric_key}")
        return

    for i, method in enumerate(methods):
        if FILTER_SPECS:
            is_match = False
            for group in FILTER_SPECS:
                if all(kw in method for kw in group):
                    is_match = True
                    break
            if not is_match: continue

        # このメソッドに含まれるすべての実験設定（初期点数など）を取得
        init_points_keys = sorted(data[method].keys())
        
        x_means, y_means = [], []
        x_stds, y_stds = [], []
        plot_points = [] # (x, y) のタプルを格納するリスト
        
        for init_pt in init_points_keys:
            # Y軸の値 (PSNR or SSIM) の平均
            y_values = data[method][init_pt][metric_key]
            if not y_values:
                continue
            y_mean = np.mean(y_values)
            y_std = np.std(y_values)
            # X軸の値の決定
            if x_axis_key == 'final':
                # 実際の最終点数の平均を使用 (Mask手法など変化する場合に対応)
                final_pts_values = data[method][init_pt]['final_points']
                x_mean = np.mean(final_pts_values)
                x_std = np.std(final_pts_values)
            else:
                # 初期点数を使用 (Baseline比較用など)
                x_mean = init_pt
                x_std = 0
            
            if MAX_PLOT_POINTS is not None and x_mean > MAX_PLOT_POINTS:
                    continue
            
            x_means.append(x_mean)
            y_means.append(y_mean)
            x_stds.append(x_std)
            y_stds.append(y_std)
        
        if x_means:
            combined = sorted(zip(x_means, y_means, x_stds, y_stds), key=lambda x: x[0])
            xs, ys, xerrs, yerrs = zip(*combined)
            marker = markers[i % len(markers)]
            if SHOW_ERROR_BARS:
                plt.errorbar(xs, ys, 
                             xerr=xerrs if x_axis_key == 'final' else None, 
                             yerr=yerrs, 
                             marker=marker, 
                             linestyle='-', 
                             linewidth=1.5,
                             capsize=3,
                             elinewidth=1.0,
                             alpha=0.8, 
                             label=method)
            else:
                plt.plot(xs, ys, marker=marker, linestyle='-', linewidth=2, label=method)

            plotted_count += 1

    if plotted_count == 0:
        print(f"Warning: No data matched filters {FILTER_KEYWORDS}. Skipping plot.")
        plt.close()
        return

    plt.xlabel("Number of Gaussian Points (Average Final)" if x_axis_key == 'final' else "Number of Initial Gaussians")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)

    if LEGEND_MODE == "outside":
        # グラフ枠外の右上に配置 (bbox_to_anchor=(x, y))
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
        # レイアウト調整 (右側が見切れないようにする)
        plt.subplots_adjust(right=0.75) 
    elif LEGEND_MODE == "inside":
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
    else:
        # "none" の場合は表示しない
        plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Saved plot: {output_file}")
    plt.close()

# ==========================================
# メイン処理 (呼び出し部分の変更)
# ==========================================
def main():
    print("Parsing NPY files...")
    data = parse_npy_logs(BASE_DIR, TARGET_ITERATION)
    
    if not data:
        print("No valid data found.")
        return

    # 1. 表データの作成と出力
    export_summary_table(data, OUTPUT_TABLE)
    
    # 1. PSNR vs Final Points
    # X軸を 'final' に指定して、実際の削減後の点数に対してプロットします
    plot_comparison(data, 'psnr', 'Average PSNR (dB)', 
                   f"PSNR vs Final Points (Iter: {TARGET_ITERATION})", 
                   "plots/comparison_psnr_vs_points.png", 
                   x_axis_key='final') # <--- ここ重要

    # 2. MS-SSIM vs Final Points
    plot_comparison(data, 'ssim', 'Average MS-SSIM', 
                   f"MS-SSIM vs Final Points (Iter: {TARGET_ITERATION})", 
                   "plots/comparison_ssim_vs_points.png", 
                   x_axis_key='final')

    # (参考) 元の Initial Points ベースの比較も残したい場合
    # plot_comparison(data, 'psnr', 'Average PSNR', 
    #                "PSNR vs Initial Points", 
    #                "plots/comparison_psnr_initial.png", x_axis_key='initial')

if __name__ == "__main__":
    main()