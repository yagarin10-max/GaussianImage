from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan
from sklearn.cluster import DBSCAN

def run_dbscan(image_tensor, eps=0.05, min_samples=5, downsample_factor=0.25, spatial_weight=1.0):
    """
    image_tensor: [C, H, W] tensor (0-1 float)
    returns: [H, W] numpy array of segment IDs
    """
    # 1. テンソルを (H*W, C) のnumpy配列に変換
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    c, h, w = image_tensor.shape
    # 色空間でクラスタリングするため、座標情報(x, y)も特徴量に加えるとより空間的にまとまる
    # ここではシンプルに色だけでDBSCANする例 (必要に応じてxyを加える)
    # img_flat = image_tensor.permute(1, 2, 0).reshape(-1, c).detach().cpu().numpy()
    small_image = F.interpolate(image_tensor.unsqueeze(0), scale_factor=downsample_factor, mode='bilinear', align_corners=False).squeeze(0)
    
    sc, sh, sw = small_image.shape
    # フラット化: [sh*sw, C]
    yy, xx = torch.meshgrid(torch.linspace(0, 1, sh), torch.linspace(0, 1, sw), indexing='ij')
    
    # [2, sh, sw]
    coords = torch.stack([xx, yy], dim=0).to(small_image.device)
    
    # 4. 特徴量の結合: Color + Weight * Coord
    # [C+2, sh, sw] -> (R, G, B, w*x, w*y)
    features_img = torch.cat([small_image, coords * spatial_weight], dim=0)
    
    # フラット化: [sh*sw, C+2]
    features_flat = features_img.permute(1, 2, 0).reshape(-1, c + 2).detach().cpu().numpy()
    
    print(f"DBSCAN running on resized image: {sh}x{sw} ({len(features_flat)} pixels) with spatial features")
    # 2. DBSCAN実行
    # eps: 色の距離の閾値 (0-1スケールなので小さめに)
    # min_samples: クラスタとみなす最小ピクセル数
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(features_flat)
    labels_small = clustering.labels_.reshape(sh, sw)
    labels_tensor = torch.from_numpy(labels_small).float().unsqueeze(0).unsqueeze(0) # [1, 1, sh, sw]
    labels_upsampled = F.interpolate(labels_tensor, size=(h, w), mode='nearest').squeeze().numpy().astype(int) # [H, W]
    # 3. ラベルを画像サイズに戻す
    # labels = clustering.labels_.reshape(h, w)
    return labels_upsampled

def init_points_by_segmentation(image, num_points):
    if image.dim() == 4:
        image = image.squeeze(0)
    print("Running DBSCAN segmentation...")
    # 1. セグメンテーション実行 (例: 色によるクラスタリングなど)
    # segments_map: [H, W] のIDマップ
    eps = 10.0 / 255.0
    segments_map = run_dbscan(image, eps=eps) 
    unique_ids = np.unique(segments_map)
    unique_ids = unique_ids[unique_ids != -1]
    xyz_list = []
    c, h, w = image.shape
    total_pixels = h * w
    points_allocated = 0
    
    for seg_id in unique_ids:
        # このセグメントのピクセル座標を取得
        ys, xs = np.where(segments_map == seg_id)
        area = len(ys)
        
        # 面積比に応じてガウシアンを配分
        n_points = int(num_points * (area / total_pixels))
        if n_points == 0: continue
            
        # セグメント内からランダムに座標をサンプリング
        indices = np.random.choice(area, n_points, replace=True)
        
        # ピクセル座標 (0~W) を Normalize座標 (-1~1) に変換
        seg_xyz = np.stack([
            (xs[indices] / (w-1)) * 2 - 1, # x
            (ys[indices] / (h-1)) * 2 - 1  # y
        ], axis=1)
        
        xyz_list.append(torch.from_numpy(seg_xyz).float())
        points_allocated += n_points
        
    print(f"Segmentation Init: Allocated {points_allocated} points based on {len(unique_ids)} segments.")
    
    # もし計算上の誤差で num_points に足りない場合はランダムで埋める
    if points_allocated < num_points:
        diff = num_points - points_allocated
        random_fill = (torch.rand(diff, 2) * 2 - 1).float()
        xyz_list.append(random_fill)
        
    return torch.cat(xyz_list, dim=0)

class GaussianImage_Cholesky(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]
        init_mode = kwargs.get("init_mode", "random")
        target_image = kwargs.get("gt_image", None)

        if init_mode == "segmentation" and target_image is not None:
            # Segmentation Initialization (AS-SMoE)
            # 画像を渡して初期座標を計算
            init_xyz = init_points_by_segmentation(target_image, self.init_num_points)
            # 初期化された座標数が num_points と完全に一致するように調整 (catの結果次第でズレる可能性があるため)
            if init_xyz.shape[0] > self.init_num_points:
                init_xyz = init_xyz[:self.init_num_points]
            # atanhを通してパラメータ化 (無限大回避のため少し縮小)
            self._xyz = nn.Parameter(torch.atanh(init_xyz.to(self.device) * (1 - 1e-4)))
        else:
            # Random Initialization (R-SMoE / GI)
            # グリッドかランダムか
            if self.init_num_points == self.H * self.W:
                yy, xx = torch.meshgrid(torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W), indexing='ij')
                grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
                self._xyz = nn.Parameter(torch.atanh(grid.to(self.device) * (1 - 1e-4)))
            else:
                self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2, device=self.device) - 0.5)))

        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        def inverse_sigmoid(x):
            return math.log(x / (1 - x))
        init_opacity = 0.1
        self._opacity = nn.Parameter(torch.ones((self.init_num_points, 1)) * inverse_sigmoid(init_opacity))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.random_colors = torch.rand(self.init_num_points, 3) # for gaussian visualization
        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self._cholesky = nn.Parameter(torch.zeros(self.init_num_points, 3))
        self.radius = (max(self.H, self.W) / math.sqrt(self.init_num_points)) * 1.2
        self.register_buffer('cholesky_bound', torch.tensor([self.radius, 0, self.radius]).view(1, 3))
        self.pruning_mode = None
        self.no_clamp = kwargs.get("no_clamp", False)

        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3)
        
        self.xyz_lr_init = 0.01
        self.xyz_lr_final = 0.00001
        self.other_lr = 0.001
        
        # パラメータグループを作成
        param_groups = [
            {'params': [self._xyz], 'lr': self.xyz_lr_init, 'name': 'xyz'},
            {'params': [self._cholesky], 'lr': self.other_lr, 'name': 'cholesky'},
            {'params': [self._features_dc], 'lr': self.other_lr, 'name': 'color'},
            {'params': [self._opacity], 'lr': 0.05, 'name': 'opacity'} # Opacityは記述がないため、GSの慣例で高めに設定するか、colorと同じにする
        ]
        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(param_groups, eps=1e-15)
        else:
            self.optimizer = Adan(param_groups, eps=1e-15)
        total_steps = kwargs.get("iterations", 10000)
        gamma = (self.xyz_lr_final / self.xyz_lr_init) ** (1.0 / total_steps)

        def lr_lambda(step):
            # stepがtotal_stepsを超えたら学習率を変えない（あるいは終了）
            if step >= total_steps:
                return (self.xyz_lr_final / self.xyz_lr_init)
            return gamma ** step

        # グループごとに適用する関数を変える
        # xyzグループ(index 0)には lr_lambda を、それ以外には 1.0 (変化なし) を適用
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=[lr_lambda, lambda s: 1.0, lambda s: 1.0, lambda s: 1.0]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    def forward(self, **kwargs):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        colors = self.get_features
        opacities = self.get_opacity
        black_bg = torch.zeros_like(self.background)
        # SMoE
        # rendered image
        numerator_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, opacities, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=black_bg, return_alpha=False)
        
        denominator_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                torch.ones_like(colors), opacities, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=black_bg, return_alpha=False)
        smoe_img = numerator_img / (denominator_img + 1e-5)

        if not self.no_clamp:
            smoe_img = torch.clamp(smoe_img, 0, 1) #[H, W, 3]
        smoe_img = smoe_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        # numerator_img = numerator_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": smoe_img, "final_opacities": opacities}

    def train_iter(self, gt_image, iterations):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        # tile_size = 16
        # pred_tiles = F.unfold(image, kernel_size=tile_size, stride=tile_size)
        # gt_tiles = F.unfold(gt_image, kernel_size=tile_size, stride=tile_size)

        # tile_l2 = torch.mean((pred_tiles - gt_tiles) ** 2, dim=1)
        
        # mean_error = tile_l2.mean().detach()
        # weights = (tile_l2.detach() / (mean_error + 1e-8)).clamp(min=0.1, max=5.0)

        # weighted_l2_loss = torch.mean(tile_l2 * weights)
        # loss += weighted_l2_loss
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        self.scheduler.step()
        return loss, psnr

    def forward_quantize(self):
        l_vqm, m_bit = 0, 16*self.init_num_points*2
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr, r_bit = 0, 0
        colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7) + render_pkg["vq_loss"]
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss, psnr

    def compress_wo_ec(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements,}

    def decompress_wo_ec(self, encoding_dict):
        xyz, feature_dc_index, quant_cholesky_elements = encoding_dict["xyz"], encoding_dict["feature_dc_index"], encoding_dict["quant_cholesky_elements"]
        means = torch.tanh(xyz.float())
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}

    def analysis_wo_ec(self, encoding_dict):
        quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
        total_bits = 0
        initial_bits, codebook_bits = 0, 0
        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        initial_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        initial_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        initial_bits += codebook_bits

        total_bits += initial_bits
        total_bits += self._xyz.numel()*16

        feature_dc_index = feature_dc_index.int().cpu().numpy()
        index_max = np.max(feature_dc_index)
        max_bit = np.ceil(np.log2(index_max)) #calculate max bit for feature_dc_index
        total_bits += feature_dc_index.size * max_bit #get_np_size(encoding_dict["feature_dc_index"]) * 8
        
        quant_cholesky_elements = quant_cholesky_elements.cpu().numpy()
        total_bits += quant_cholesky_elements.size * 6 #cholesky bits 

        position_bits = self._xyz.numel()*16
        cholesky_bits, feature_dc_bits = 0, 0
        cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        cholesky_bits += quant_cholesky_elements.size * 6
        feature_dc_bits += codebook_bits
        feature_dc_bits += feature_dc_index.size * max_bit

        bpp = total_bits/self.H/self.W
        position_bpp = position_bits/self.H/self.W
        cholesky_bpp = cholesky_bits/self.H/self.W
        feature_dc_bpp = feature_dc_bits/self.H/self.W
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp}

    def compress(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        quant_cholesky_elements, cholesky_elements = self.cholesky_quantizer.compress(self._cholesky)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_cholesky_elements": quant_cholesky_elements, 
            "feature_dc_bitstream":[feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique], 
            "cholesky_bitstream":[cholesky_compressed, cholesky_histogram_table, cholesky_unique]}

    def decompress(self, encoding_dict):
        xyz = encoding_dict["xyz"]
        num_points, device = xyz.size(0), xyz.device
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = encoding_dict["feature_dc_bitstream"]
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = encoding_dict["cholesky_bitstream"]
        feature_dc_index = decompress_matrix_flatten_categorical(feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique, num_points*2, (num_points, 2))
        quant_cholesky_elements = decompress_matrix_flatten_categorical(cholesky_compressed, cholesky_histogram_table, cholesky_unique, num_points*3, (num_points, 3))
        feature_dc_index = torch.from_numpy(feature_dc_index).to(device).int() #[800, 2]
        quant_cholesky_elements = torch.from_numpy(quant_cholesky_elements).to(device).float() #[800, 3]

        means = torch.tanh(xyz.float())
        cholesky_elements = self.cholesky_quantizer.decompress(quant_cholesky_elements)
        cholesky_elements = cholesky_elements + self.cholesky_bound
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means, cholesky_elements, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}
   
    def analysis(self, encoding_dict):
        quant_cholesky_elements, feature_dc_index = encoding_dict["quant_cholesky_elements"], encoding_dict["feature_dc_index"]
        cholesky_compressed, cholesky_histogram_table, cholesky_unique = compress_matrix_flatten_categorical(quant_cholesky_elements.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())  
        cholesky_lookup = dict(zip(cholesky_unique, cholesky_histogram_table.astype(np.float64) / np.sum(cholesky_histogram_table).astype(np.float64)))
        feature_dc_lookup = dict(zip(feature_dc_unique, feature_dc_histogram_table.astype(np.float64) / np.sum(feature_dc_histogram_table).astype(np.float64)))

        total_bits = 0
        initial_bits, codebook_bits = 0, 0
        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        initial_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        initial_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        initial_bits += get_np_size(cholesky_histogram_table) * 8
        initial_bits += get_np_size(cholesky_unique) * 8 
        initial_bits += get_np_size(feature_dc_histogram_table) * 8
        initial_bits += get_np_size(feature_dc_unique) * 8  
        initial_bits += codebook_bits

        total_bits += initial_bits
        total_bits += self._xyz.numel()*16
        total_bits += get_np_size(cholesky_compressed) * 8
        total_bits += get_np_size(feature_dc_compressed) * 8

        position_bits = self._xyz.numel()*16
        cholesky_bits, feature_dc_bits = 0, 0
        cholesky_bits += self.cholesky_quantizer.scale.numel()*torch.finfo(self.cholesky_quantizer.scale.dtype).bits
        cholesky_bits += self.cholesky_quantizer.beta.numel()*torch.finfo(self.cholesky_quantizer.beta.dtype).bits
        cholesky_bits += get_np_size(cholesky_histogram_table) * 8
        cholesky_bits += get_np_size(cholesky_unique) * 8   
        cholesky_bits += get_np_size(cholesky_compressed) * 8
        feature_dc_bits += codebook_bits
        feature_dc_bits += get_np_size(feature_dc_histogram_table) * 8
        feature_dc_bits += get_np_size(feature_dc_unique) * 8  
        feature_dc_bits += get_np_size(feature_dc_compressed) * 8

        bpp = total_bits/self.H/self.W
        position_bpp = position_bits/self.H/self.W
        cholesky_bpp = cholesky_bits/self.H/self.W
        feature_dc_bpp = feature_dc_bits/self.H/self.W
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp,}
