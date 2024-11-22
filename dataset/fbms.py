import os
from glob import glob
from PIL import Image
import torch
import torchvision as tv


class TestFBMS(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.video_list = sorted(os.listdir(os.path.join(root, 'JPEGImages')))
        self.to_tensor = tv.transforms.ToTensor()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        img_dir = os.path.join(self.root, 'JPEGImages', video_name)
        flow_dir = os.path.join(self.root, 'JPEGFlows', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', video_name)
        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        # generate testing snippets
        imgs = []
        flows = []
        for i in range(len(img_list)):
            img = Image.open(img_list[i]).convert('RGB')
            imgs.append(self.to_tensor(img))
        for i in range(len(flow_list)):
            flow = Image.open(flow_list[i]).convert('RGB')
            flows.append(self.to_tensor(flow))

        # gather all frames
        imgs = torch.stack(imgs, dim=0)
        flows = torch.stack(flows, dim=0)
        return {'imgs': imgs, 'flows': flows, 'video_name': video_name, 'files': mask_list}
