import torch
import torchvision
from PIL import Image

from archs.Flickerformer_arch import Flickerformer
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
    "--model_type", type=str, default="Flickerformer"
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
