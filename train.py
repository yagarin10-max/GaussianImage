import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import wandb

class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
        start_mask_training: int = 0,
        stop_mask_training: int = 50000,
        use_wandb: bool = False,
        wandb_project: str = "GaussianImage",
        wandb_entity: str = None,
        reg_type: str = "kl",
        target_sparsity: float = 0.7,
        lambda_reg: float = 0.005,
        init_mask_logit: float = 2.0,
        use_ema: bool = False,
        use_score: bool = False,
        no_clamp: bool = False,
        temp_init: float = 0.5,
        temp_final: float = 0.5,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device)

        self.init_mask = None
        if args.mask_dataset is not None:
            mask_dir = Path(args.mask_dataset)            
            mask_path = mask_dir / f"{image_path.stem}_binary.png"

            if mask_path.exists():
                print(f"Loading mask from: {mask_path}")
                mask_tensor = image_path_to_tensor(mask_path).to(self.device)
                if mask_tensor.shape[1] > 1:
                    self.init_mask = mask_tensor.mean(dim=1, keepdim=True)
                else:
                    self.init_mask = mask_tensor
            else:
                print(f"Warning: Mask file not found for {image_path.name}. Checked: {mask_path}")
        # -------------------------------

        self.num_points = num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs

        # self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
        if model_name == "GaussianImage_Cholesky_wMask":
            suffix = ""
            if use_ema: suffix += "_ema"
            if use_score: suffix += "_score"
            if no_clamp: suffix += "_noclp"
            if args.temp_init == args.temp_final:
                temp_str = f"T{args.temp_init}"
            else:
                temp_str = f"T{args.temp_init}-{args.temp_final}"
            suffix += f"_{temp_str}"

            folder_name = f"maskGI_Ch_{reg_type}_tgt{target_sparsity}_lam{lambda_reg}_init{init_mask_logit}_{args.iterations}_{num_points}{suffix}"
        else:
            suffix = ""
            if no_clamp: suffix += "_noclp"
            folder_name = f"{model_name}_{args.iterations}_{num_points}{suffix}"
            
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{folder_name}/{self.image_name}")
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"{self.image_name}_{folder_name}",
                config={
                    "model": model_name,
                    "image": self.image_name,
                    "iterations": iterations,
                    "num_points": num_points,
                    "lr": args.lr,
                    "start_mask": start_mask_training,
                    "stop_mask": stop_mask_training,
                    "reg_type": reg_type,
                    "target_sparsity": target_sparsity,
                    "lambda_reg": lambda_reg,
                    "init_mask_logit": init_mask_logit,
                    "use_ema": use_ema,
                    "use_score": use_score,
                    "no_clamp": no_clamp,
                    "temp_init": temp_init,
                    "temp_final": temp_final,
                },
            )
        if model_name == "GaussianImage_Cholesky_wMask":
            from gaussianimage_cholesky_wMask import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False, start_mask_training=start_mask_training, stop_mask_training=stop_mask_training,
                reg_type=reg_type, target_sparsity=target_sparsity, lambda_reg=lambda_reg, init_mask_logit=init_mask_logit,
                use_ema=use_ema, use_score=use_score, no_clamp=no_clamp, temp_init=temp_init, temp_final=temp_final).to(self.device)
        
        elif model_name == "GaussianImage_Cholesky":
            from gaussianimage_cholesky import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False, no_clamp=no_clamp, init_mask=self.init_mask, match_mask_points=args.match_mask_points).to(self.device)

        elif model_name == "GaussianImage_RS":
            from gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device) 

        elif model_name == "3DGS":
            from gaussiansplatting_3d import Gaussian3D
            self.gaussian_model = Gaussian3D(loss_type="Fusion2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

        if args.match_mask_points and self.init_mask is not None:
                new_points = self.gaussian_model._xyz.shape[0]
                print(f"Num points updated from {self.num_points} to {new_points} based on mask.")
                self.num_points = new_points

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def train(self):     
        psnr_list, iter_list, loss_list = [], [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter(self.gt_image, iter)
            psnr_list.append(psnr)
            iter_list.append(iter)
            loss_list.append(loss.item())

            if self.use_wandb and iter % 100 == 0:
                log_data = {
                    "train/loss": loss.item(),
                    "train/psnr": psnr,
                    "iter": iter,
                }
                if hasattr(self.gaussian_model, "_mask_logits"):
                    with torch.no_grad():
                        probs = torch.sigmoid(self.gaussian_model._mask_logits)
                        sparsity_hard = (probs > 0.5).float().mean().item()
                        sparsity_soft = probs.mean().item()
                        log_data["train/sparsity_hard"] = sparsity_hard
                        log_data["train/sparsity_soft"] = sparsity_soft
                        current_points = self.gaussian_model._xyz.shape[0]
                        log_data["train/num_points_active"] = int(current_points * sparsity_hard)
                wandb.log(log_data)
            
            if self.use_wandb and (iter % 5000 == 0):
                with torch.no_grad():
                    render_pkg = self.gaussian_model.forward(pruning_mode=self.gaussian_model.pruning_mode)
                    img_tensor = render_pkg["render"].clamp(0, 1)
                    img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    gauss_img_tensor = render_pkg["gauss_render"].clamp(0, 1)
                    gauss_img_np = gauss_img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    alpha_img_tensor = render_pkg["alpha_map"]
                    alpha_img_np = alpha_img_tensor.squeeze().cpu().numpy()
                    vmax = 6.0
                    norm_alpha = np.clip(alpha_img_np, 0, vmax) / vmax
                    colors = [
                        (0.0 / vmax, "black"),
                        (1.0 / vmax, "lime"),
                        (3.0 / vmax, "orange"),
                        (6.0 / vmax, "darkred"),
                    ]
                    custom_cmap = mcolors.LinearSegmentedColormap.from_list("densitiy_fixed_cmap", colors)
                    alpha_heatmap = custom_cmap(norm_alpha)[:, :, :3]

                    points_np = self.gaussian_model.xys.detach().cpu().numpy()
                    final_opacities = render_pkg["final_opacities"].squeeze().cpu().numpy()
                    valid_indices = final_opacities > 0.001
                    valid_points = points_np[valid_indices]

                    # --- 2. 描画用ヘルパー関数 (ここに追加！) ---
                    def overlay_points_on_image(bg_image, points):
                        h, w, _ = bg_image.shape
                        fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
                        
                        # 画像を表示
                        ax.imshow(bg_image)
                        # 点をプロット (色はlimeで見やすく)
                        ax.scatter(points[:, 0], points[:, 1], s=1, c='lime', marker='o', alpha=1.0, linewidths=0)
                        
                        ax.axis('off')
                        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                        plt.margins(0,0)
                        
                        fig.canvas.draw()
                        data = np.array(fig.canvas.buffer_rgba())
                        plt.close(fig)
                        return data[:, :, :3]

                    # --- 3. 3つの画像すべてに適用 ---
                    img_with_points = overlay_points_on_image(img_np, valid_points)
                    # gauss_with_points = overlay_points_on_image(gauss_img_np, valid_points)
                    # heatmap_with_points = overlay_points_on_image(alpha_heatmap, valid_points)
                    wandb.log({
                        "render_image": [wandb.Image(img_np, caption=f"Iter {iter}")],
                        "gauss_image": [wandb.Image(gauss_img_np, caption=f"Iter {iter}")],
                        "alpha_heatmap": [wandb.Image(alpha_heatmap, caption=f"Alpha(0-3) Iter {iter}")],
                        "render_with_points": [wandb.Image(img_with_points, caption=f"Render+Pts {iter}")],
                        # "gauss_with_points": [wandb.Image(gauss_with_points, caption=f"Gauss+Pts {iter}")],
                        # "heatmap_with_points": [wandb.Image(heatmap_with_points, caption=f"Heatmap+Pts {iter}")],
                        "iter": iter,
                    }
                    )
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        if hasattr(self.gaussian_model, 'prune_points'):
            print("Pruning points...")
            self.gaussian_model.prune_points(threshold=0.5)
        

        psnr_value, ms_ssim_value, num_points_final = self.test(pruning_mode=self.gaussian_model.pruning_mode)
        if self.use_wandb:
            wandb.run.summary["final_num_gaussians"] = num_points_final
            wandb.run.summary["final_psnr"] = psnr_value
            wandb.run.summary["final_ms_ssim"] = ms_ssim_value

        if self.use_wandb:
            wandb.finish()

        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time,
        "initial_points": self.num_points, "final_points": num_points_final})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def test(self, pruning_mode="None"):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model(pruning_mode=pruning_mode)
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        num_points_final = self.gaussian_model._xyz.shape[0]
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}, Final_points:{:d}".format(psnr, ms_ssim_value, num_points_final))
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = self.image_name + "_fitting.png" 
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value, num_points_final

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--start_mask_training", type=int, default=0, help="Iteration to start soft mask training"
    )
    parser.add_argument(
        "--stop_mask_training", type=int, default=50000, help="Iteration to stop soft mask training and switch to hard mask"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="GaussianImage", help="Wandb project name")
    parser.add_argument("--reg_type", type=str, default="kl", help="Regularization type for mask training: kl, l1, l1sq")
    parser.add_argument("--target_sparsity", type=float, default=0.7, help="Target sparsity for KL divergence regularization")
    parser.add_argument("--lambda_reg", type=float, default=0.005, help="Regularization weight for mask training")
    parser.add_argument("--init_mask_logit", type=float, default=2.0, help="Initial mask logit value")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA for mask logit")
    parser.add_argument("--use_score", action="store_true", help="Use score for masking")
    parser.add_argument("--no_clamp", action="store_true", help="Disable clamping in rendering")
    parser.add_argument("--temp_init", type=float, default=0.5, help="Initial temperature for Gumbel-Softmax")
    parser.add_argument("--temp_final", type=float, default=0.5, help="End/min temperature for Gumbel-Softmax")
    parser.add_argument("--mask_dataset", type=str, default=None, help="Path to the binary mask dataset folder")
    parser.add_argument("--match_mask_points", action="store_true", help="If True, ignore --num_points and use the exact number of valid points in binary mask.")

    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
    
    suffix = ""
    if args.use_ema: suffix += "_ema"
    if args.use_score: suffix += "_score"
    if args.no_clamp: suffix += "_noclp"
    
    if args.model_name == "GaussianImage_Cholesky_wMask":
        folder_name = f"maskGI_Ch_{args.reg_type}_tgt{args.target_sparsity}_lam{args.lambda_reg}_init{args.init_mask_logit}_{args.iterations}_{args.num_points}{suffix}"
    else:
        folder_name = f"{args.model_name}_{args.iterations}_{args.num_points}{suffix}"
    
    logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{folder_name}"))

    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0
    target_images = []
    if args.data_name == "kodak":
        target_images = [Path(args.dataset) / f'kodim{i+1:02}.png' for i in range(24)]
    elif args.data_name == "test":
        target_images = [Path(args.dataset) / f'test{i+1:02}.png' for i in range(2)]
    elif args.data_name == "kodak_small":
        target_images = [Path(args.dataset) / f'kodim{i+1:02}.png' for i in range(1)]    
    elif args.data_name == "DIV2K_valid_LRX2":
        target_images = [Path(args.dataset) / f'{i+1:04}x2.png' for i in range(800, 900)] # rangeは適宜
    elif args.data_name == "binary":
        dataset_path = Path(args.dataset)
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        target_images = sorted([
            p for p in dataset_path.iterdir() 
            if p.suffix.lower() in valid_extensions
        ])
        print(f"Found {len(target_images)} images in {args.dataset}")

    target_images = target_images[0:1] ### for debug

    image_length = len(target_images)
    if image_length == 0:
        print("Error: No images found!")
        return
    for image_path in target_images:
        print(f"Processing image: {image_path.name}")
        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
            iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path, 
            start_mask_training=args.start_mask_training, stop_mask_training=args.stop_mask_training, use_wandb=args.use_wandb, wandb_project=args.wandb_project,
            reg_type=args.reg_type, target_sparsity=args.target_sparsity, lambda_reg=args.lambda_reg, init_mask_logit=args.init_mask_logit,
            use_ema=args.use_ema, use_score=args.use_score, no_clamp=args.no_clamp, temp_init=args.temp_init, temp_final=args.temp_final)
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps))    

if __name__ == "__main__":
    main(sys.argv[1:])
