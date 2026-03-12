import torch
import torchvision
from PIL import Image

from basicsr.archs.Burstormer_arch import Burstormer
from basicsr.archs.HDRTransformer_arch import HDRTransformer
from basicsr.archs.HINT_arch import HINT
from basicsr.archs.Restormer_arch import Restormer
from basicsr.archs.Restormer_vis_arch import Restormer_vis
from basicsr.archs.AFUNet_arch import AFUNet
from basicsr.archs.SAFNet_arch import SAFNet
from basicsr.archs.SCTNet_arch import SCTNet
from basicsr.archs.DarkIR_arch import DarkIR
from basicsr.archs.oppo_arch import OPPO
from basicsr.archs.flickerformer_arch import FlickerNet
from basicsr.archs.SAFNet_version3_arch import SAFNet_3
from basicsr.archs.SAFNet_version2_arch import SAFNet_2
from basicsr.archs.SAFNet_version4_arch import SAFNet_4
from basicsr.archs.SAFNet_version5_arch import SAFNet_5
from basicsr.archs.SAFNet_version4_vis_arch import SAFNet_4_vis
from basicsr.archs.SAFNet_version5_vis_arch import SAFNet_5_vis
from basicsr.archs.SAFNet_version6_vis_arch import SAFNet_6_vis
from basicsr.archs.FFT_Retinexformer_arch import FFT_Retinexformer
from basicsr.archs.SAFNet_version6_arch import SAFNet_6
from basicsr.archs.SAFNet_version7_arch import SAFNet_7
from basicsr.archs.SAFNet_version8_arch import SAFNet_8
from basicsr.archs.SAFNet_version9_arch import SAFNet_9
from basicsr.archs.SAFNet_version4_fft_arch import SAFNet_4_FFT
from basicsr.archs.SAFNet_version4_channel_att_arch import SAFNet_4_channel_att
from basicsr.archs.Flickerformer_arch import Uformer_Cross
# from basicsr.archs.Flickerformer_SCAM_arch import Flickerformer_SCAM
from basicsr.archs.Flickerformer_SCAM2_arch import Flickerformer_SCAM2
from basicsr.archs.Flickerformer_SCAM3_arch import Flickerformer_SCAM3
from basicsr.archs.Flickerformer_SCAM4_arch import Flickerformer_SCAM4
from basicsr.archs.Flickerformer_SCAM5_arch import Flickerformer_SCAM5
from basicsr.archs.Flickerformer_SCAM6_arch import Flickerformer_SCAM6
from basicsr.archs.Flickerformer_SCAM7_arch import Flickerformer_SCAM7
from basicsr.archs.Flickerformer_SCAM8_arch import Flickerformer_SCAM8
from basicsr.archs.Flickerformer_SCAM9_arch import Flickerformer_SCAM9
from basicsr.archs.Flickerformer_SCAM10_arch import Flickerformer_SCAM10
from basicsr.archs.Flickerformer_SCAM11_arch import Flickerformer_SCAM11
from basicsr.archs.Flickerformer_SCAM12_arch import Flickerformer_SCAM12
from basicsr.archs.Flickerformer_SCAM13_arch import Flickerformer_SCAM13
from basicsr.archs.Flickerformer_SCAM14_arch import Flickerformer_SCAM14
from basicsr.archs.Flickerformer_SCAM14_vis_arch import Flickerformer_SCAM14_vis
from basicsr.archs.Flickerformer_SCAM15_arch import Flickerformer_SCAM15
# from basicsr.archs.Flickerformer_SCAM16_arch import Flickerformer_SCAM16
from basicsr.archs.Flickerformer_SCAM16_light_arch import Flickerformer_light_SCAM16
from basicsr.archs.Flickerformer_SCAM16_light_phase_arch import Flickerformer_light_phase_SCAM16
from basicsr.archs.Flickerformer_SCAM16_vis_arch import Flickerformer_SCAM16_vis
from basicsr.archs.Flickerformer_SCAM17_arch import Flickerformer_SCAM17
from basicsr.archs.Flickerformer_SCAM18_arch import Flickerformer_SCAM18
from basicsr.archs.Flickerformer_SCAM19_arch import Flickerformer_SCAM19
from basicsr.archs.Flickerformer_SCAM20_arch import Flickerformer_SCAM20
from basicsr.archs.Flickerformer_SCAM21_arch import Flickerformer_SCAM21
from basicsr.archs.Flickerformer_SCAM22_arch import Flickerformer_SCAM22
from basicsr.archs.Flickerformer_SCAM23_arch import Flickerformer_SCAM23
from basicsr.archs.Flickerformer_SCAM24_arch import Flickerformer_SCAM24
from basicsr.archs.Flickerformer_SCAM25_arch import Flickerformer_SCAM25
from basicsr.archs.Flickerformer_SCAM25_vis_arch import Flickerformer_SCAM25_vis
from basicsr.archs.Flickerformer_SCAM26_arch import Flickerformer_SCAM26
from basicsr.archs.Flickerformer_SCAM27_arch import Flickerformer_SCAM27
from basicsr.archs.Flickerformer_SCAM28_arch import Flickerformer_SCAM28
from basicsr.archs.Flickerformer_SCAM29_arch import Flickerformer_SCAM29
from basicsr.archs.Flickerformer_SCAM30_arch import Flickerformer_SCAM30
from basicsr.archs.Flickerformer_SCAM31_arch import Flickerformer_SCAM31
from basicsr.archs.Flickerformer_SCAM34_arch import Flickerformer_SCAM34
from basicsr.archs.Flickerformer_SCAM35_arch import Flickerformer_SCAM35
from basicsr.archs.Flickerformer_SCAM36_arch import Flickerformer_SCAM36
from basicsr.archs.Flickerformer_SCAM37_arch import Flickerformer_SCAM37
from basicsr.archs.Flickerformer_SCAM38_arch import Flickerformer_SCAM38
from basicsr.archs.Flickerformer_SCAM39_arch import Flickerformer_SCAM39
from basicsr.archs.Flickerformer_SCAM41_arch import Flickerformer_SCAM41
import argparse
import torchvision.transforms as transforms
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="/home/notebook/data/personal/S9059954/BurstDeflicker/dataset/BurstFlicker-S/test-resize/input")
parser.add_argument("--output", type=str, default="/home/notebook/data/personal/S9059954/BurstDeflicker/result")
parser.add_argument(
    "--model_path", type=str, default="/home/notebook/data/personal/S9059954/BurstDeflicker/weights/"
)
parser.add_argument(
    "--model_type", type=str, default="Restormer_vis"
)

args = parser.parse_args()
images_path = os.path.join(args.input)
result_path = args.output
pretrain_dir = args.model_path


model_type = args.model_type

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def load_params(model_path):
    full_model = torch.load(model_path)
    if "params_ema" in full_model:
        return full_model["params_ema"]
    elif "params" in full_model:
        return full_model["params"]
    else:
        return full_model

def demo(images_path, output_path,  pretrain_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(images_path)
    torch.cuda.empty_cache()
    model = globals()[model_type]().cuda()
    
    model.load_state_dict(load_params(pretrain_path))

    to_tensor = transforms.ToTensor()

    resize = transforms.Resize((512,512))  # The output should in the shape of 128X

    for root, _, files in os.walk(images_path):
        files.sort()
        for file in files:
            if file.endswith(".png") or file.endswith(".JPG"):
            # Get current image index
                idx = files.index(file)

                # Read current image
                img1 = Image.open(os.path.join(root, file)).convert("RGB")
                img1_ori = to_tensor(img1).cuda()
                resize2org = transforms.Resize((img1.size[1], img1.size[0]))
                # img1 = img1.resize((img1.size[0] // 2, img1.size[1] // 2))
                # print(1)
                # img1 = torch.nn.functiona l.interpolate(img1, scale_factor=0.5)

                # Handle getting img2 and img3 based on position in sequence
                # if idx + 2 < len(files):
                #     # Normal case - get next 2 images
                #     img2 = Image.open(os.path.join(root, files[idx + 1])).convert("RGB")
                #     img3 = Image.open(os.path.join(root, files[idx + 2])).convert("RGB")
                # elif idx + 1 < len(files):
                #     # Only 1 image after current, get 1 before for img3
                #     img2 = Image.open(os.path.join(root, files[idx + 1])).convert("RGB")
                #     img3 = Image.open(os.path.join(root, files[idx - 1])).convert("RGB")
                # else:
                #     # No images after current, get 2 before
                #     img2 = Image.open(os.path.join(root, files[idx - 2])).convert("RGB")
                #     img3 = Image.open(os.path.join(root, files[idx - 1])).convert("RGB")

                # 取中间帧当作base
                if idx == 0:
                    # Normal case - get next 2 images
                    img2 = Image.open(os.path.join(root, files[idx + 1])).convert("RGB")
                    img3 = Image.open(os.path.join(root, files[idx + 2])).convert("RGB")
                elif idx == len(files)-1:
                    # Only 1 image after current, get 1 before for img3
                    img2 = Image.open(os.path.join(root, files[idx - 1])).convert("RGB")
                    img3 = Image.open(os.path.join(root, files[idx - 2])).convert("RGB")
                else:
                    # No images after current, get 2 before
                    img2 = Image.open(os.path.join(root, files[idx - 1])).convert("RGB")
                    img3 = Image.open(os.path.join(root, files[idx + 1])).convert("RGB")

                img1 = resize(to_tensor(img1).cuda())
                img2 = resize(to_tensor(img2).cuda())
                img3 = resize(to_tensor(img3).cuda())
                # img2 = to_tensor(img2).cuda()
                # img1 = to_tensor(img1).cuda()
                # img3 = to_tensor(img3).cuda()

                model.eval()
                with torch.no_grad():

                    output_img= model(torch.cat([img2, img1, img3], dim=0).unsqueeze(0))
                    output_img = img1_ori.unsqueeze(0) + resize2org(output_img - img1)
                    output_file = os.path.join(output_path, str(idx) + ".png")
                    torchvision.utils.save_image(output_img, output_file)

def get_subfolders(folder_path):
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    return subfolders

folders = get_subfolders(images_path)

model_files = [f for f in os.listdir(pretrain_dir) if f.startswith('net_g_') and f.endswith('.pth')]
model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
pretrain_path = os.path.join(pretrain_dir, model_files[-1])

# pretrain_path = pretrain_dir
# model_files =  pretrain_dir.split("/")[-1]

output_model = pretrain_path.split("/")[-3]

for path in folders:
    demo(path, os.path.join(result_path,output_model,model_files[-1],path.split("/")[-1]), pretrain_path)
    # demo(path, os.path.join(result_path,model_type,model_files,path.split("/")[-1]), pretrain_path)
