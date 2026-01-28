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
    ["kl", "noclp"],      # グループ2: "kl" と "tgt0.7" の両方を含むならOK
    ["kl", "ema"]
    # ["l1"]     # 必要ならグループ3を追加...
]

def format_method_name(prefix, suffix):
    """
    prefix: _50000_xxxxx より前の部分 (例: maskGI_..._init1.0)
    suffix: _50000_xxxxx より後の部分 (例: _noclp)
    """
    name = ""
    
    # --- 1. ベースの手法名を決定 ---
    if "GaussianImage_Cholesky" in prefix and "wMask" not in prefix:
        name = "Baseline"
    elif "maskGI" in prefix:
        # パラメータ抽出 (adaなどもあれば対応)
        # maskGI_Ch_(ada_)?kl... のようにadaがある場合も考慮
        match = re.search(r"maskGI_Ch_(?:ada_)?([^_]+)_tgt([^_]+)_lam([^_]+)_init([^_]+)", prefix)
        if match:
            reg, tgt, lam, init_val = match.group(1), match.group(2), match.group(3), match.group(4)
            name = f"Mask ({reg}, tgt{tgt}, λ{lam}, init{init_val})"
            if "ada" in prefix:
                name += " [Ada]"
        else:
            name = prefix # マッチしない場合はそのまま
    else:
        name = prefix

    # --- 2. サフィックス (noclp, ema, score) をタグ付け ---
    # suffixには "_noclp" や "_ema_score" などが入ってくる
    if "noclp" in suffix or "noclp" in prefix: # prefixに含まれるケースも考慮
        name += " [No Clip]"
    
    if "ema" in suffix:
        name += " [EMA]"
        
    if "score" in suffix:
        name += " [Score]"

    return name


def parse_npy_logs(base_dir, target_iter):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # 検索パターン: イテレーション番号が含まれるディレクトリ全て
    search_pattern = os.path.join(base_dir, f"*_{target_iter}_*")
    exp_dirs = glob.glob(search_pattern)
    
    print(f"Found {len(exp_dirs)} experiment directories matching iter {target_iter}.")

    for exp_dir in exp_dirs:
        dir_name = os.path.basename(exp_dir)
        
        # 正規表現の変更:
        # ^(.*)            -> Group 1: プレフィックス (手法名 + パラメータ)
        # _{target_iter}_  -> イテレーション (区切り文字)
        # (\d+)            -> Group 2: 初期点数
        # (.*)$            -> Group 3: サフィックス (_noclp, _ema など。無い場合は空文字)
        match = re.search(rf"^(.*)_{target_iter}_(\d+)(.*)$", dir_name)
        
        if not match:
            continue
            
        prefix = match.group(1)
        init_points = int(match.group(2))
        suffix = match.group(3) # "_noclp" など、あるいは空文字
        
        # 表示名を作成
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
    figsize = (12, 6) if LEGEND_MODE == "outside" else (10, 6)
    plt.figure(figsize=figsize)
    
    methods = sorted(data.keys())
    plotted_count = 0
    
    # カラーサイクル (Baselineなどは同じ色にしたい場合、ここで工夫も可能ですが
    # 基本は自動割り当てで、線種で区別するのが分かりやすいです)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, method in enumerate(methods):
        # --- フィルタリング ---
        if FILTER_SPECS:
            is_match = False
            for group in FILTER_SPECS:
                # 検索対象を「表示名(method)」にするか「元のディレクトリ名」にするか迷いますが
                # format_method_nameで "No Clip" 等を入れているので、method検索でOKです。
                # 小文字にして検索することで "noclp" でも "No Clip" でもヒットしやすくします。
                method_lower = method.lower()
                
                # グループ内のキーワードがすべて含まれているか (AND条件)
                # キーワード側も小文字化して比較
                if all(kw.lower() in method_lower for kw in group):
                    is_match = True
                    break
            if not is_match:
                continue

        # --- スタイル決定ロジック ---
        linestyle = '-' # デフォルト: 実線
        marker = markers[i % len(markers)]
        alpha = 0.8
        
        # [No Clip] が含まれていれば点線にする
        if "[No Clip]" in method:
            linestyle = '--' 
            alpha = 0.6 # 少し薄くする
            
        # [EMA] が含まれていれば一点鎖線にする（お好みで）
        # if "[EMA]" in method:
        #    linestyle = '-.'

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
        
        if x_means:
            combined = sorted(zip(x_means, y_means, x_stds, y_stds), key=lambda x: x[0])
            xs, ys, xerrs, yerrs = zip(*combined)
            
            if SHOW_ERROR_BARS:
                plt.errorbar(
                    xs, ys, 
                    xerr=xerrs if x_axis_key == 'final' else None,
                    yerr=yerrs, 
                    marker=marker, 
                    linestyle=linestyle, # ここで点線/実線を適用
                    linewidth=1.5,
                    capsize=3,
                    elinewidth=1.0,
                    alpha=alpha,
                    label=method
                )
            else:
                plt.plot(xs, ys, marker=marker, linestyle=linestyle, linewidth=2, label=method)
            
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
    print("Parsing NPY files with advanced suffix support...")
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