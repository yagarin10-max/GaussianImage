import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================================
# 設定
# ==========================================
BASE_DIR = "checkpoints/kodak_small/"
TARGET_ITERATION = "50000"  # 固定するIteration数

# 保存するグラフのファイル名
OUTPUT_PSNR_AVG = "plots/plot_psnr_average.png"
OUTPUT_SSIM_AVG = "plots/plot_ssim_average.png"
OUTPUT_PSNR_ALL = "plots/plot_psnr_all_images.png"
OUTPUT_SSIM_ALL = "plots/plot_ssim_all_images.png"

# ==========================================
# データ読み込みと解析
# ==========================================
def parse_logs(base_dir, target_iter):
    # データ構造: metrics[num_gaussians][image_name] = {'psnr': float, 'ssim': float}
    metrics = defaultdict(dict)
    
    # ディレクトリ名のパターン: GaussianImage_Cholesky_50000_{num_gaussians}
    # 正規表現でディレクトリを検索
    dir_pattern = os.path.join(base_dir, f"GaussianImage_Cholesky_{target_iter}_*")
    experiment_dirs = glob.glob(dir_pattern)
    
    print(f"Found {len(experiment_dirs)} experiment directories.")

    for exp_dir in experiment_dirs:
        # ディレクトリ名から初期ガウシアン数を抽出
        dir_name = os.path.basename(exp_dir)
        match_dir = re.search(rf"GaussianImage_Cholesky_{target_iter}_(\d+)", dir_name)
        
        if not match_dir:
            continue
            
        num_gaussians = int(match_dir.group(1))
        
        # 各画像ディレクトリ (kodim01, kodim02...) を探索
        image_dirs = glob.glob(os.path.join(exp_dir, "kodim*"))
        
        for img_dir in image_dirs:
            img_name = os.path.basename(img_dir)
            train_txt_path = os.path.join(img_dir, "train.txt")
            
            if os.path.exists(train_txt_path):
                try:
                    with open(train_txt_path, 'r') as f:
                        content = f.read()
                        # 正規表現で数値を抽出
                        # Test PSNR:31.5981, MS_SSIM:0.989548
                        match_metrics = re.search(r"Test PSNR:([\d\.]+),\s*MS_SSIM:([\d\.]+)", content)
                        if match_metrics:
                            psnr = float(match_metrics.group(1))
                            ssim = float(match_metrics.group(2))
                            
                            metrics[num_gaussians][img_name] = {
                                'psnr': psnr,
                                'ssim': ssim
                            }
                except Exception as e:
                    print(f"Error reading {train_txt_path}: {e}")
    
    return metrics

# ==========================================
# グラフ描画関数
# ==========================================
def plot_graph(x_values, y_values_dict, y_label, title, output_file, show_individual=False):
    plt.figure(figsize=(10, 6))
    
    # x_values (ガウシアン数) でソートするためのインデックス
    sorted_indices = np.argsort(x_values)
    x_sorted = np.array(x_values)[sorted_indices]
    
    # 全画像の平均を計算するためのリスト
    all_y_matrix = [] # (num_points, num_images)
    
    # 画像ごとのデータを収集
    # y_values_dict: {img_name: [val_at_x1, val_at_x2...]} 形式に変換済みと仮定
    
    image_names = list(y_values_dict.keys())
    
    for img_name in image_names:
        # x_sortedに対応するyの値を取り出す
        y_list = []
        for x in x_sorted:
            # データが存在しない場合はNaNなどを入れる処理が必要だが、今回はある前提で進める
            y_list.append(y_values_dict[img_name].get(x, None))
        
        # Noneが含まれていない場合のみプロットに使用
        if None not in y_list:
            all_y_matrix.append(y_list)
            if show_individual:
                plt.plot(x_sorted, y_list, alpha=0.3, linewidth=1, label='_nolegend_') # 個別線は薄く
    
    # 平均値の計算とプロット
    if all_y_matrix:
        all_y_matrix = np.array(all_y_matrix)
        y_mean = np.mean(all_y_matrix, axis=0)
        
        plt.plot(x_sorted, y_mean, color='blue', linewidth=2.5, marker='o', label='Average')
        
        # 最大値・最小値の幅を塗りつぶす（オプション：ばらつきが見やすくなります）
        if show_individual:
            y_max = np.max(all_y_matrix, axis=0)
            y_min = np.min(all_y_matrix, axis=0)
            plt.fill_between(x_sorted, y_min, y_max, color='blue', alpha=0.1)

    plt.xlabel("Number of Initial Gaussians")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_file)
    print(f"Saved: {output_file}")
    plt.close()

# ==========================================
# メイン処理
# ==========================================
def main():
    data = parse_logs(BASE_DIR, TARGET_ITERATION)
    
    if not data:
        print("No data found. Check directory structure.")
        return

    # プロット用にデータを整理
    # x軸: ガウシアン数
    x_values = list(data.keys())
    
    # 画像ごとのy軸データ辞書を作成
    # psnr_data[img_name] = {num_gaussians: value}
    all_images = set()
    for num_g in data:
        all_images.update(data[num_g].keys())
    
    psnr_data_by_img = defaultdict(dict)
    ssim_data_by_img = defaultdict(dict)
    
    for num_g in x_values:
        for img in all_images:
            if img in data[num_g]:
                psnr_data_by_img[img][num_g] = data[num_g][img]['psnr']
                ssim_data_by_img[img][num_g] = data[num_g][img]['ssim']

    # 1. PSNR 平均グラフ
    plot_graph(x_values, psnr_data_by_img, "PSNR", 
               f"PSNR vs Number of Gaussians (Iter: {TARGET_ITERATION}) - Average", 
               OUTPUT_PSNR_AVG, show_individual=False)

    # 2. MS-SSIM 平均グラフ
    plot_graph(x_values, ssim_data_by_img, "MS-SSIM", 
               f"MS-SSIM vs Number of Gaussians (Iter: {TARGET_ITERATION}) - Average", 
               OUTPUT_SSIM_AVG, show_individual=False)
               
    # 3. PSNR 全画像重ね合わせグラフ
    plot_graph(x_values, psnr_data_by_img, "PSNR", 
               f"PSNR vs Number of Gaussians - All Images", 
               OUTPUT_PSNR_ALL, show_individual=True)

    # 4. MS-SSIM 全画像重ね合わせグラフ
    plot_graph(x_values, ssim_data_by_img, "MS-SSIM", 
               f"MS-SSIM vs Number of Gaussians - All Images", 
               OUTPUT_SSIM_ALL, show_individual=True)

if __name__ == "__main__":
    main()