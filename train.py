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
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device)

        self.num_points = num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"{self.image_name}_{model_name}_{iterations}_{num_points}",
                config={
                    "model": model_name,
                    "image": self.image_name,
                    "iterations": iterations,
                    "num_points": num_points,
                    "lr": args.lr,
                    "start_mask": start_mask_training,
                    "stop_mask": stop_mask_training,
                },
            )
        if model_name == "GaussianImage_Cholesky_wMask":
            from gaussianimage_cholesky_wMask import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False, start_mask_training=start_mask_training, stop_mask_training=stop_mask_training).to(self.device)
        
        elif model_name == "GaussianImage_Cholesky":
            from gaussianimage_cholesky import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)

        elif model_name == "GaussianImage_RS":
            from gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device) 

        elif model_name == "3DGS":
            from gaussiansplatting_3d import Gaussian3D
            self.gaussian_model = Gaussian3D(loss_type="Fusion2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

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

                    wandb.log({
                        "render_image": [wandb.Image(img_np, caption=f"Iter {iter}")],
                        "gauss_image": [wandb.Image(gauss_img_np, caption=f"Iter {iter}")],
                        "alpha_heatmap": [wandb.Image(alpha_heatmap, caption=f"Alpha(0-3) Iter {iter}")],
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
        
        if self.use_wandb:
            wandb.finish()

        psnr_value, ms_ssim_value = self.test(pruning_mode=self.gaussian_model.pruning_mode)
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def test(self, pruning_mode="None"):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model(pruning_mode=pruning_mode)
        mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out["render"].float().squeeze(0))
            name = self.image_name + "_fitting.png" 
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value

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

    logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0
    if args.data_name == "kodak":
        image_length, start = 24, 0
    elif args.data_name == "kodak_small":
        image_length, start = 3, 0
    elif args.data_name == "DIV2K_valid_LRX2":
        image_length, start = 100, 800
    for i in range(start, start+image_length):
        if (args.data_name == "kodak") or (args.data_name == "kodak_small"):
            image_path = Path(args.dataset) / f'kodim{i+1:02}.png'
        elif args.data_name == "DIV2K_valid_LRX2":
            image_path = Path(args.dataset) /  f'{i+1:04}x2.png'

        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
            iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path, 
            start_mask_training=args.start_mask_training, stop_mask_training=args.stop_mask_training, use_wandb=args.use_wandb, wandb_project=args.wandb_project)
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
