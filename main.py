import argparse
import traceback
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import skimage
import torch_fidelity
import tqdm
import lpips
import logging
import yaml
import sys
import os
import torch
import numpy as np
from guided_diffusion.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=1234, help="Set different seeds for diverse results")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--deg", type=str, required=True, help="Degradation")
    parser.add_argument("--path_y", type=str, required=True, help="Path of the test dataset.")
    parser.add_argument("--sigma_y", type=float, default=0., help="sigma_y")
    parser.add_argument("--deg_scale", type=float, default=0., help="deg_scale")
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument('--subset_start', type=int, default=-1)
    parser.add_argument('--subset_end', type=int, default=-1)
    parser.add_argument("--verbose", type=str, default="info", help="Verbose level: info | debug | warning | critical")
    parser.add_argument("-i", "--image_folder", type=str, default="images", help="The folder name of samples")
    parser.add_argument("--eta", type=float,nargs='+', default=0.85, help="Eta")
    parser.add_argument('--ntrials', type=int, default=1, help="The number of trials for average")
    parser.add_argument('--NFE', type=int, default=100, help="The number of evaluations")

    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
    args.image_folder = os.path.join(
        args.exp, "image_samples", args.image_folder
    )
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def cal_metrics(path):
    orig_path = os.path.join(path, f"ref")
    generated_path = os.path.join(path, f"sample")
    N = len(os.listdir(generated_path))
    assert N == 1000
    SSIM_sum = 0
    PSNR_sum = 0
    LPIPS_sum = 0
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    print('calculating PSNR, SSIM & LPIPS')
    with torch.no_grad():
        for k in tqdm.tqdm(range(N)):
            source_path = os.sep.join([orig_path, 'orig_{}.png'.format(k)])
            source_image = skimage.io.imread(source_path)/255.00
            denoise_path = os.sep.join([generated_path, '{}_0.png'.format(k)])
            generated_image = skimage.io.imread(denoise_path)/255.0
            SSIM = ssim(source_image, generated_image, data_range=generated_image.max() - generated_image.min(), channel_axis=-1)
            SSIM_sum += SSIM
            PSNR = psnr(source_image, generated_image)
            PSNR_sum += PSNR
            source_image = source_image * 2 - 1
            generated_image = generated_image * 2 - 1
            LPIPS = loss_fn_vgg(torch.tensor(source_image).permute(2,0,1).to(torch.float32).cuda(), torch.tensor(generated_image).permute(2,0,1).to(torch.float32).cuda())
            LPIPS_sum += LPIPS[0,0,0,0]
        print('Average LPIPS: {}'.format(LPIPS_sum/N))
        print('calculating KID & FID')
        Results = torch_fidelity.calculate_metrics(input1=orig_path, input2=generated_path, kid=True, fid=True)
        print('PSNR:{}, SSIM:{}, LPIPS:{}, KID:{}, FID:{}'.format(PSNR_sum/N, SSIM_sum/N, LPIPS_sum/N, Results['kernel_inception_distance_mean'], Results['frechet_inception_distance']))

def main():
    args, config = parse_args_and_config()
    try:
        # runner = Diffusion(args, config)
        # runner.sample()
        cal_metrics(args.image_folder)
    except Exception:
        logging.error(traceback.format_exc())
    return 0

if __name__ == "__main__":
    sys.exit(main())
