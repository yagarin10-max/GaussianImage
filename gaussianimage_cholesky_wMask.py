from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan

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
        self.start_mask_training = kwargs.get("start_mask_training", 0)
        self.stop_mask_training = kwargs.get("stop_mask_training", 50000)
        
        if self.init_num_points == self.H * self.W:
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, self.H), torch.linspace(-1, 1, self.W), indexing='ij')
            grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
            self._xyz = nn.Parameter(torch.atanh(grid * (1 - 1e-4))) # avoid exactly -1 or 1
        else:
            self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        
        self._cholesky = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))
        self.init_mask_logit = kwargs.get("init_mask_logit", 2.0)
        self._mask_logits = nn.Parameter(torch.ones(self.init_num_points, 1) * self.init_mask_logit) 
        self.random_colors = torch.rand(self.init_num_points, 3) # for gaussian visualization
        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(3))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.reg_type = kwargs.get("reg_type", "kl")  # "kl" or "l1" or "l1sq"
        self.lambda_reg = kwargs.get("lambda_reg", 0.005)
        self.target_sparsity = kwargs.get("target_sparsity", 0.7)
        self.pruning_mode = None
        self.mask_scores_ema = None
        self.ema_decay = 0.99
        self.use_ema = kwargs.get("use_ema", False)
        self.use_score = kwargs.get("use_score", False)
        self.no_clamp = kwargs.get("no_clamp", False)

        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.cholesky_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)

    def _gumbel_sigmoid(self, input, temperature=0.5, hard=False, eps = 1e-10):
        """
        A gumbel-sigmoid nonlinearity with gumbel(0,1) noize
        In short, it's a function that mimics #[a>0] indicator where a is the logit
        Explaination and motivation: https://arxiv.org/abs/1611.01144
        """
        with torch.no_grad():
        # generate a random sample from the uniform distribution
            uniform1 = torch.rand(input.size(), device=input.device)
            uniform2 = torch.rand(input.size(), device=input.device)
            gumbel_noise = -torch.log(torch.log(uniform1 + eps)/torch.log(uniform2 + eps) + eps) #.cuda()

        reparam = (input + gumbel_noise)/temperature
    #         print(reparam)
        y_soft = torch.sigmoid(reparam)     
        if hard:
            # Straight through.
            y_hard = (y_soft > 0.5).float()
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    @torch.no_grad()
    def prune_points(self, threshold=0.5):
        probs = torch.sigmoid(self._mask_logits)
        valid_indices = (probs > threshold).squeeze()
        
        # パラメータを更新して削減
        self._xyz = nn.Parameter(self._xyz[valid_indices])
        self._cholesky = nn.Parameter(self._cholesky[valid_indices])
        self._features_dc = nn.Parameter(self._features_dc[valid_indices])
        self._mask_logits = nn.Parameter(self._mask_logits[valid_indices])
        self._opacity = self._opacity[valid_indices] # bufferなのでParameterではない
        
        # 点数を更新
        self.init_num_points = self._xyz.shape[0]
        
        # Quantizer等の再初期化が必要な場合はここで行う
        # self.cholesky_quantizer._init_data(self._cholesky) 
        
        print(f"Pruned points: {probs.shape[0]} to {self.init_num_points} points.")

    def calculate_importance_score(self):
        # 1. 実際のCholesky要素を取得 (Boundを加算したもの)
        # shape: [N, 3]
        cholesky = self.get_cholesky_elements

        # 2. 面積係数を計算 (対角成分の積)
        # 行列式 |L| = L00 * L11
        # 要素0と要素2を取り出して掛ける
        L00 = cholesky[:, 0]
        L11 = cholesky[:, 2]
        
        # これがガウシアンの物理的な「面積」に比例する値です
        # abs()は念のためですが、boundのおかげで正になっているはずです
        area = (L00 * L11).abs().view(-1, 1)  # shape: [N, 1]

        # 3. 不透明度を取得 (Pruningモードに合わせる)
        opacities = self._opacity
        
        # 4. 重要度スコア = 不透明度 × 面積
        importance_score = opacities * area
        
        return importance_score

    def get_current_temperature(self, iterations):
        # マスク学習の開始と終了
        start_iter = self.start_mask_training
        end_iter = self.stop_mask_training
        
        if iterations < start_iter:
            return 1.0 # 使わないのでなんでもいい
        
        # 進行度 (0.0 -> 1.0)
        progress = (iterations - start_iter) / (end_iter - start_iter)
        progress = max(0.0, min(1.0, progress))
        
        # 指数減衰させる (例: 5.0 -> 0.1)
        temp_start = 5.0
        temp_end = 0.1
        # log spaceでの線形補間（指数的な減少）が一般的によく効きます
        current_temp = temp_start * (temp_end / temp_start) ** progress
        
        return current_temp

    def forward(self, pruning_mode=None, temperature=1.0): # Puning mode: "None", "hard", "soft"
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)
        
        colors = self.get_features
        opacities = self._opacity.clone()
        if self.use_score:
            score = self.calculate_importance_score().detach()
            mask_input = self._mask_logits * score
        else:
            mask_input = self._mask_logits
        mask = None
        if pruning_mode =="soft":
            mask = self._gumbel_sigmoid(mask_input, temperature=temperature, hard=False)
        elif pruning_mode =="hard":
            mask = self._gumbel_sigmoid(mask_input, temperature=temperature, hard=True)
        elif pruning_mode =="deterministic":
            mask = (torch.sigmoid(self._mask_logits) > 0.5).float()

        if mask is not None:
            opacities = opacities * mask

        # rendered image
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, opacities, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        if not self.no_clamp:
            out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        # gaussian visualization
        geom_colors = self.random_colors.to(self.xys.device) * 0.5 # x0.5 to make it visually dark to avoid too much saturation
        gauss_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                geom_colors, opacities, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        gauss_img = torch.clamp(gauss_img, 0, 1) #[H, W, 3]
        gauss_img = gauss_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        # alpha map visualization
        ones_color = torch.ones_like(colors)
        alpha_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                ones_color, opacities, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_alpha = alpha_img.mean(dim=-1, keepdim=True)
        out_alpha = out_alpha.view(-1, self.H, self.W, 1).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img, "gauss_render": gauss_img, "alpha_map": out_alpha, "final_opacities": opacities}

    def train_iter(self, gt_image, iterations):
        #TODO optimize the loss calculation
        if iterations < self.start_mask_training:
            self.pruning_mode = None
        elif iterations < self.stop_mask_training:
            self.pruning_mode = "soft"

            if self.use_ema:
                with torch.no_grad():
                    current_probs = torch.sigmoid(self._mask_logits)
                    if self.mask_scores_ema is None:
                        self.mask_scores_ema = current_probs.detach().clone()
                    else:
                        self.mask_scores_ema = self.ema_decay * self.mask_scores_ema + (1 - self.ema_decay) * current_probs
        elif iterations == self.stop_mask_training and self.use_ema: 
            self.pruning_mode = "deterministic"   
            with torch.no_grad():
                final_mask = self.mask_scores_ema > 0.5
                self._mask_logits[final_mask] = 10.0  # large positive value to ensure sigmoid ~ 1
                self._mask_logits[~final_mask] = -10.0  # large negative value to ensure sigmoid ~ 0
                print("Finalized masks for deterministic pruning.")
        else: 
            # pruning_mode = "hard"
            self.pruning_mode = "deterministic"
        # current_temp = self.get_current_temperature(iterations)
        render_pkg = self.forward(pruning_mode=self.pruning_mode, temperature=0.5) #current_temp)
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)

        tile_size = 16
        pred_tiles = F.unfold(image, kernel_size=tile_size, stride=tile_size)
        gt_tiles = F.unfold(gt_image, kernel_size=tile_size, stride=tile_size)

        # tile_l2 = torch.mean((pred_tiles - gt_tiles) ** 2, dim=1)
        
        # mean_error = tile_l2.mean().detach()
        # weights = (tile_l2.detach() / (mean_error + 1e-8)).clamp(min=0.1, max=5.0)

        # weighted_l2_loss = torch.mean(tile_l2 * weights)
        # loss += weighted_l2_loss

        if self.pruning_mode != None and self.pruning_mode != "deterministic" and iterations >= self.start_mask_training:
            mask_probs = torch.sigmoid(self._mask_logits)

            # === KL Divergence ===
            if self.reg_type == "kl":
                current_rho = torch.mean(mask_probs)
                current_rho = torch.clamp(current_rho, 1e-5, 1.0 - 1e-5)
                target_rho = torch.clamp(torch.tensor(self.target_sparsity), 1e-5, 1.0 - 1e-5)

                loss_kl = (target_rho * torch.log(target_rho / current_rho) + (1 - target_rho) * torch.log((1 - target_rho) / (1 - current_rho)))
                loss += self.lambda_reg * loss_kl

            elif self.reg_type == "ada_kl":
                H, W = self.H, self.W
                kl_loss = self.calc_adaptive_sparsity_scatter(gt_tiles, mask_probs, self.xys, H, W, tile_size, sparsity_min=self.target_sparsity, sparsity_max=0.9)
                loss += self.lambda_reg * kl_loss

            # === L1 Regularization ===
            elif self.reg_type == "l1":
                loss += self.lambda_reg * torch.mean(mask_probs)
            
            elif self.reg_type == "l1sq":
                loss += self.lambda_reg * torch.mean(mask_probs)**2

        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        self.scheduler.step()
        return loss, psnr

    def calc_adaptive_sparsity_scatter(self, gt_tiles, mask_probs, xys, H, W, tile_size, sparsity_min, sparsity_max):
        tile_complexity = torch.var(gt_tiles, dim=1).squeeze(0).detach() # [num_tiles]
        
        # 正規化して0~1の範囲にする (バッチ内の最大分散で割るなど)
        tile_complexity = torch.clamp(tile_complexity, min=1e-6)
        tile_complexity_log = torch.log(tile_complexity)
        c_min = tile_complexity_log.min()
        c_max = torch.quantile(tile_complexity_log, 0.95)

        normalized_complexity = (tile_complexity_log - c_min) / (c_max - c_min + 1e-5)
        normalized_complexity = torch.clamp(normalized_complexity, 0.0, 1.0)
        target_rho = sparsity_min + (sparsity_max - sparsity_min) * normalized_complexity
        target_rho = target_rho.detach() # Targetは定数扱い
        # タイルのグリッド数
        num_tiles_x = W // tile_size
        num_tiles_y = H // tile_size
        total_tiles = num_tiles_x * num_tiles_y
        
        # 座標をデタッチ (正則化で点を移動させないため。移動させたいなら外す)
        xys_detached = xys.detach()

        # 各点がどのタイルに落ちるか計算
        tile_idx_x = (xys_detached[:, 0] / tile_size).long()
        tile_idx_y = (xys_detached[:, 1] / tile_size).long()

        # 画面外の点は無視するためのマスク
        valid_points = (tile_idx_x >= 0) & (tile_idx_x < num_tiles_x) & \
                       (tile_idx_y >= 0) & (tile_idx_y < num_tiles_y)
        
        # 有効な点のみ抽出
        valid_mask_probs = mask_probs[valid_points].squeeze(-1) # [M]
        valid_idx_x = tile_idx_x[valid_points]
        valid_idx_y = tile_idx_y[valid_points]
        
        # 1次元のタイルインデックスに変換 (0 ~ total_tiles-1)
        linear_tile_idx = valid_idx_y * num_tiles_x + valid_idx_x # [M]

        # 集計用のTensor準備
        tile_mask_sum = torch.zeros(total_tiles, device=mask_probs.device)
        tile_point_count = torch.zeros(total_tiles, device=mask_probs.device)

        # Scatter Add: インデックスに従って値を加算
        # マスク値の合計
        tile_mask_sum.scatter_add_(0, linear_tile_idx, valid_mask_probs)
        # 点の数の合計 (すべて1を加算)
        tile_point_count.scatter_add_(0, linear_tile_idx, torch.ones_like(valid_mask_probs))

        # 平均の計算 (0除算回避)
        # 点が1つもないタイルは、勾配0でよいので、current=targetにしてロスを0にするなどの処理が必要
        # ここでは count > 0 の部分だけ計算し、0の部分はtargetと同じ値で埋める戦略をとります
        
        current_rho = torch.zeros_like(target_rho)
        has_points_mask = tile_point_count > 0
        
        current_rho[has_points_mask] = tile_mask_sum[has_points_mask] / tile_point_count[has_points_mask]
        
        # 点がないタイルはLoss計算から除外するため、targetと同じ値を入れておく (log(1)=0になる)
        # または単にマスクする
        current_rho[~has_points_mask] = target_rho[~has_points_mask]

        current_rho = torch.clamp(current_rho, 1e-5, 1.0 - 1e-5)
        target_rho = torch.clamp(target_rho, 1e-5, 1.0 - 1e-5)

        min_len = min(current_rho.shape[0], target_rho.shape[0])
        current_rho = current_rho[:min_len]
        target_rho = target_rho[:min_len]

        kl_per_tile = (target_rho * torch.log(target_rho / current_rho) + 
                       (1 - target_rho) * torch.log((1 - target_rho) / (1 - current_rho)))

        return torch.mean(kl_per_tile)

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
