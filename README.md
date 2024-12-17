# RealFlow

This is the official PyTorch implementation of our paper:

> **Transforming Static Images Using Generative Models for Video Salient Object Detection**, *arXiv 2024*\
> Suhwan Cho, Minhyeok Lee, Jungho Lee, Sangyoun Lee\
> Link: [[arXiv]](https://arxiv.org/pdf/2411.13975)

<img src="https://github.com/user-attachments/assets/4466342a-4eed-4b55-a13d-f85f0972db9f" width=800>

You can also find other related papers at [awesome-video-object-segmentation](https://github.com/suhwan-cho/awesome-video-object-segmentation).


## Abstract
Leveraging large-scale image datasets is a common strategy in video processing tasks. However, motion-guided 
approaches are limited in this regard, as spatial distortions cannot effectively simulate realistic motion dynamics. 
In this study, we demonstrate that **image-to-video generation models** can generate realistic optical flows by appropriately 
transforming static images. Our simulated video dataset, **DUTSv3**, offers valuable potential for future research.



## Preparation
1\. Download 
[DUTS](http://saliencydetection.net/duts/#org3aad434), 
[DAVIS16](https://davischallenge.org/davis2016/code.html),
[DAVSOD](https://github.com/DengPingFan/DAVSOD),
[FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets),
and [ViSal](https://github.com/shenjianbing/ViSalDataset?tab=readme-ov-file)
from the official websites.

2\. Estimate and save optical flow maps from the videos using [RAFT](https://github.com/princeton-vl/RAFT).

3\. For DUTS, simulate optical flow maps using [Stable Video Diffusion](https://github.com/Stability-AI/generative-models).

4\. For convenience, I also provide the pre-processed
[DUTSv3](https://drive.google.com/file/d/107qMPd2w8BDrmywbEcYYzniScAXX94NS/view?usp=drive_link),
[DAVIS16](https://drive.google.com/file/d/10qzxi6bNkM44SXYjjX2BDra7qQMUdbot/view?usp=drive_link),
[FBMS](https://drive.google.com/file/d/1arrBPuhjnwlKlgmCZveUsKw5uForr4JN/view?usp=drive_link),
[DAVSOD](https://drive.google.com/file/d/1zJxuejRJOXD9Wd7D5jH3_KkoUikvpYH3/view?usp=drive_link),
and [ViSal](https://drive.google.com/file/d/19FNPm-kPS_x_IoPBHSHaLW18L3xMK1VN/view?usp=drive_link).

5\. Replace dataset paths in "run.py" file with your dataset paths.


## Training
1\. Open the "run.py" file.

2\. Specify the model version.

3\. Verify the training settings.

4\. Start **RealFlow** training!
```
python run.py --train
```


## Testing
1\. Open the "run.py" file.

2\. Specify the model version.

3\. Choose a pre-trained model.

4\. Start **RealFlow** testing!
```
python run.py --test
```

## Attachments
[pre-trained model (mitb0)](https://drive.google.com/file/d/1dE7hlUMvzNT7PCoA3MXydh9OOW04kHii/view?usp=drive_link)\
[pre-trained model (mitb1)](https://drive.google.com/file/d/1zgrhrEtDE8HDbQK5RVDwYiF-m4UZuHEL/view?usp=drive_link)\
[pre-trained model (mitb2)](https://drive.google.com/file/d/1p89m8kRGuXLghe9jO7yX03ByMMXayyTl/view?usp=drive_link)\
[pre-computed results](https://drive.google.com/file/d/1KnZH9hlfs7vQGIs9oXMmJ-WUBRM06JyO/view?usp=drive_link)


## Note
Code and models are only available for non-commercial research purposes.\
If you have any questions, please feel free to contact me :)
```
E-mail: suhwanx@gmail.com
```
