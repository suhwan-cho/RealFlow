import os
import random
from glob import glob
from PIL import Image
import torch
import torchvision as tv
import torchvision.transforms.functional as TF


class TrainDAVSOD(torch.utils.data.Dataset):
    def __init__(self, root, clip_n):
        self.root = root
        self.video_list = sorted(os.listdir(os.path.join(root, 'JPEGImages')))
        self.clip_n = clip_n
        self.to_tensor = tv.transforms.ToTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        video_name = random.choice(self.video_list)
        img_dir = os.path.join(self.root, 'JPEGImages', video_name)
        flow_dir = os.path.join(self.root, 'JPEGFlows', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', video_name)
        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        flow_list = sorted(glob(os.path.join(flow_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        # select training frame
        all_frames = list(range(len(img_list)))
        frame_id = random.choice(all_frames)
        img = Image.open(img_list[frame_id]).convert('RGB')
        flow = Image.open(flow_list[frame_id]).convert('RGB')
        mask = Image.open(mask_list[frame_id]).convert('L')

        # resize to 512p
        img = img.resize((512, 512), Image.BICUBIC)
        flow = flow.resize((512, 512), Image.BICUBIC)
        mask = mask.resize((512, 512), Image.BICUBIC)

        # joint flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            flow = TF.hflip(flow)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            flow = TF.vflip(flow)
            mask = TF.vflip(mask)

        # convert formats
        imgs = self.to_tensor(img).unsqueeze(0)
        flows = self.to_tensor(flow).unsqueeze(0)
        masks = self.to_tensor(mask).unsqueeze(0)
        masks = (masks > 0.5).long()
        return {'imgs': imgs, 'flows': flows, 'masks': masks}


class TestDAVSOD(torch.utils.data.Dataset):
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
