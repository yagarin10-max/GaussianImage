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
            # init = match.group(4) # 必要ならinitも表示
            return f"Mask ({reg}, tgt{tgt}, λ{lam})"
            
    return raw_name

def parse_npy_logs(base_dir, target_iter):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # ディレクトリ検索
    search_pattern = os.path.join(base_dir, f"*_{target_iter}_*")
    exp_dirs = glob.glob(search_pattern)
    
    print(f"Found {len(exp_dirs)} experiment directories matching iter {target_iter}.")

    for exp_dir in exp_dirs:
        dir_name = os.path.basename(exp_dir)
        
        # 正規表現で「手法名」と「初期点数」を分離
        # 末尾の _{target_iter}_{num_points} を基準に分割
        match = re.search(rf"^(.*)_{target_iter}_(\d+)$", dir_name)
        
        if not match:
            continue
            
        raw_method_name = match.group(1)
        init_points = int(match.group(2)) # ディレクトリ名から初期点数を取得
        
        # 表示名を整形
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
# プロット関数 (X軸を動的に計算するように修正)
# ==========================================
def plot_comparison(data, metric_key, y_label, title, output_file, x_axis_key='initial'):
    """
    x_axis_key: 'initial' なら初期点数(固定)をX軸に、
                'final' なら実際の最終点数(平均)をX軸に使用する
    """
    plt.figure(figsize=(10, 6))
    
    methods = sorted(data.keys())
    
    if not methods:
        print(f"No data found for metric: {metric_key}")
        return

    for method in methods:
        # このメソッドに含まれるすべての実験設定（初期点数など）を取得
        init_points_keys = sorted(data[method].keys())
        
        plot_points = [] # (x, y) のタプルを格納するリスト
        
        for init_pt in init_points_keys:
            # Y軸の値 (PSNR or SSIM) の平均
            y_values = data[method][init_pt][metric_key]
            if not y_values:
                continue
            y_mean = np.mean(y_values)
            
            # X軸の値の決定
            if x_axis_key == 'final':
                # 実際の最終点数の平均を使用 (Mask手法など変化する場合に対応)
                final_pts_values = data[method][init_pt]['final_points']
                x_val = np.mean(final_pts_values)
            else:
                # 初期点数を使用 (Baseline比較用など)
                x_val = init_pt
            
            plot_points.append((x_val, y_mean))
        
        # 線を綺麗に引くために、X軸の値でソートする
        # (点数が減るとX順序が入れ替わる可能性があるため)
        plot_points.sort(key=lambda p: p[0])
        
        if plot_points:
            xs, ys = zip(*plot_points)
            
            # マーカー設定
            linestyle = '-'
            marker = 'o'
            if "mask" in method.lower() or "Mask" in method:
                marker = 's' # Mask手法は四角
            
            plt.plot(xs, ys, marker=marker, linestyle=linestyle, linewidth=2, label=method)

    plt.xlabel("Number of Gaussian Points (Average Final)" if x_axis_key == 'final' else "Number of Initial Gaussians")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
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

    # 1. PSNR vs Final Points (これが欲しい図です！)
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