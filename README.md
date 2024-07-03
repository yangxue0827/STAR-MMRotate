# STAR: A First-Ever Dataset and A Large-Scale Benchmark for Scene Graph Generation in Large-Size Satellite Imagery

The official implementation of the oriented object detection part of the paper "[STAR: A First-Ever Dataset and A Large-Scale Benchmark for Scene Graph Generation in Large-Size Satellite Imagery](https://arxiv.org/abs/2406.09410)".

## ⭐️ Highlights

**TL;DR:** We propose STAR, the first large-scale dataset for scene graph generation in large-size VHR SAI. Containing more than `210,000` objects and over `400,000` triplets across `1,273` complex scenarios globally.

https://private-user-images.githubusercontent.com/29257168/339049597-2d027f2c-8911-45ba-b4dd-7f95111465a9.mp4

## 📌 Abstract

Scene graph generation (SGG) in satellite imagery (SAI) benefits promoting understanding of geospatial scenarios from perception to cognition. In SAI, objects exhibit great variations in scales and aspect ratios, and there exist rich relationships between objects (even between spatially disjoint objects), which makes it attractive to holistically conduct SGG in large-size very-high-resolution (VHR) SAI. However, there lack such SGG datasets. Due to the complexity of large-size SAI, mining triplets $<$subject, relationship, object$>$ heavily relies on long-range contextual reasoning. Consequently, SGG models designed for small-size natural imagery are not directly applicable to large-size SAI. This paper constructs a large-scale dataset for SGG in large-size VHR SAI with image sizes ranging from 512 × 768 to 27,860 × 31,096 pixels, named STAR (Scene graph generaTion in lArge-size satellite imageRy), encompassing over 210K objects and over 400K triplets. To realize SGG in large-size SAI, we propose a context-aware cascade cognition (CAC) framework to understand SAI regarding object detection (OBD), pair pruning and relationship prediction for SGG. We also release a SAI-oriented SGG toolkit with about 30 OBD and 10 SGG methods which need further adaptation by our devised modules on our challenging STAR dataset. **The dataset and toolkit are available at: https://linlin-dev.github.io/project/STAR**.

<p align="center">
<img src="demo/star.jpg" alt="scatter" width="98%"/> 
</p>

## 🛠️ Usage

More instructions on installation, pretrained models, training and evaluation, please refer to [MMRotate 0.3.4](README_en.md).
  
- Clone this repo:

  ```bash
  git clone https://github.com/yangxue0827/STAR-MMRotate
  cd STAR-MMRotate/
  ```

- Create a conda virtual environment and activate it:
  
  ```bash
  conda create -n STAR-MMRotate python=3.8 -y
  conda activate STAR-MMRotate
  ```

- Install Pytorch:

  ```bash
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  ```

- Install requirements:

  ```bash
  pip install openmim
  mim install mmcv-full
  mim install mmdet
  
  cd mmrotate
  pip install -r requirements/build.txt
  pip install -v -e .

  pip install timm
  pip install ipdb

  # Optional, only for G-Rep
  git clone git@github.com:KinglittleQ/torch-batch-svd.git
  cd torch-batch-svd/
  python setup.py install
  ```

## 🚀 Released Models

### Oriented Object Detection

|  Detector  | mAP | Configs | Download | Note |
| :--------: |:---:|:-------:|:--------:|:----:|
| Deformable DETR | 17.1 | [deformable_detr_r50_1x_star](configs/ars_detr/deformable_detr_r50_1x_star.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/deformable_detr_r50_1x_star.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/deformable_detr_r50_1x_star-fe862bb3.pth?download=true) |
| ARS-DETR | 28.1 | [dn_arw_arm_arcsl_rdetr_r50_1x_star](configs/ars_detr/dn_arw_arm_arcsl_rdetr_r50_1x_star.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/dn_arw_arm_arcsl_rdetr_r50_1x_star.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/dn_arw_arm_arcsl_rdetr_r50_1x_star-cbb34897.pth?download=true) |
| RetinaNet | 21.8 | [rotated_retinanet_hbb_r50_fpn_1x_star_oc](configs/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_star_oc.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_retinanet_hbb_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_retinanet_hbb_r50_fpn_1x_star_oc-3ec35d77.pth?download=true) |
| ATSS | 20.4 | [rotated_atss_hbb_r50_fpn_1x_star_oc](configs/rotated_atss/rotated_atss_hbb_r50_fpn_1x_star_oc.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_atss_hbb_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_atss_hbb_r50_fpn_1x_star_oc-f65f07c2.pth?download=true) | 
|  KLD  |  25.0  |   [rotated_retinanet_hbb_kld_r50_fpn_1x_star_oc](configs/kld/rotated_retinanet_hbb_kld_r50_fpn_1x_star_oc.py)  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_retinanet_hbb_kld_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_retinanet_hbb_kld_r50_fpn_1x_star_oc-343a0b83.pth?download=true) |
|  GWD  |  25.3  |   [rotated_retinanet_hbb_gwd_r50_fpn_1x_star_oc](configs/gwd/rotated_retinanet_hbb_gwd_r50_fpn_1x_star_oc.py)  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_retinanet_hbb_gwd_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_retinanet_hbb_gwd_r50_fpn_1x_star_oc-566d2398.pth?download=true) |
| KFIoU |  25.5  |   [rotated_retinanet_hbb_kfiou_r50_fpn_1x_star_oc](configs/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_star_oc.py)  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_retinanet_hbb_kfiou_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_retinanet_hbb_kfiou_r50_fpn_1x_star_oc-198081a6.pth?download=true) |
| DCFL | 29.0 | [dcfl_r50_fpn_1x_star_le135](configs/dcfl/dcfl_r50_fpn_1x_star_le135.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/dcfl_r50_fpn_1x_star_le135.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/dcfl_r50_fpn_1x_star_le135-a5945790.pth?download=true) |
| R<sup>3</sup>Det | 23.7 | [r3det_r50_fpn_1x_star_oc](configs/r3det/r3det_r50_fpn_1x_star_oc.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/r3det_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/r3det_r50_fpn_1x_star_oc-c8c4a5e5.pth?download=true) |
| S2A-Net | 27.3 | [s2anet_r50_fpn_1x_star_le135](configs/s2anet/s2anet_r50_fpn_1x_star_le135.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/s2anet_r50_fpn_1x_star_le135.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/s2anet_r50_fpn_1x_star_le135-42887a81.pth?download=true) |
| FCOS  |  28.1  |   [rotated_fcos_r50_fpn_1x_star_le90](configs/rotated_fcos/rotated_fcos_r50_fpn_1x_star_le90.py)  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_fcos_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_fcos_r50_fpn_1x_star_le90-a579fbf7.pth?download=true) | 
| CSL | 27.4 | [rotated_fcos_csl_gaussian_r50_fpn_1x_star_le90](configs/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_fcos_csl_gaussian_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_fcos_csl_gaussian_r50_fpn_1x_star_le90-6ab9a42a.pth?download=true) | 
| PSC | 30.5 | [rotated_fcos_psc_r50_fpn_1x_star_le90](configs/psc/rotated_fcos_psc_r50_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_fcos_psc_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_fcos_psc_r50_fpn_1x_star_le90-7acce1be.pth?download=true) |
| H2RBox-v2 | 27.3 | [h2rbox_v2p_r50_fpn_1x_star_le90](configs/h2rbox_v2p/h2rbox_v2p_r50_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/h2rbox_v2p_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/h2rbox_v2p_r50_fpn_1x_star_le90-25409050.pth?download=true) |
| RepPoints  | 19.7 | [rotated_reppoints_r50_fpn_1x_star_oc](configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_star_oc.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_reppoints_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_reppoints_r50_fpn_1x_star_oc-7a6c59b9.pth?download=true) |
| CFA | 25.1 | [cfa_r50_fpn_1x_star_le135](configs/cfa/cfa_r50_fpn_1x_star_le135.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/cfa_r50_fpn_1x_star_le135.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/cfa_r50_fpn_1x_star_le135-287f6b84.pth?download=true) |
| Oriented RepPoints  |  27.0  |   [oriented_reppoints_r50_fpn_1x_star_le135](configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_star_le135.py)  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/oriented_reppoints_r50_fpn_1x_star_le135.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/oriented_reppoints_r50_fpn_1x_star_le135-06389ea6.pth?download=true) | |
| G-Rep | 26.9 | [g_reppoints_r50_fpn_1x_star_le135](configs/g_reppoints/g_reppoints_r50_fpn_1x_star_le135.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/g_reppoints_r50_fpn_1x_star_le135.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/g_reppoints_r50_fpn_1x_star_le135-ec243141.pth?download=true) |
| SASM  |  28.2  |   [sasm_reppoints_r50_fpn_1x_star_oc](configs/sasm_reppoints/sasm_reppoints_r50_fpn_1x_star_oc.py)  |  [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/sasm_reppoints_r50_fpn_1x_star_oc.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/sasm_reppoints_r50_fpn_1x_star_oc-4f1ca558.pth?download=true) | [p_bs=2](https://github.com/yangxue0827/STAR-MMRotate/blob/05c0064cbcd5c44437321b50e1d2d4ee9b4445db/mmrotate/models/detectors/single_stage_crop.py#L310) |
| Faster RCNN | 32.6 | [rotated_faster_rcnn_r50_fpn_1x_star_le90](configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/rotated_faster_rcnn_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/rotated_faster_rcnn_r50_fpn_1x_star_le90-9a832bc2.pth?download=true) |
| Gliding Vertex | 30.7 | [gliding_vertex_r50_fpn_1x_star_le90](configs/gliding_vertex/gliding_vertex_r50_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/gliding_vertex_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/gliding_vertex_r50_fpn_1x_star_le90-5c0bc879.pth?download=true) |
| Oriented RCNN | 33.2 | [oriented_rcnn_r50_fpn_1x_star_le90](configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/oriented_rcnn_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/oriented_rcnn_r50_fpn_1x_star_le90-0b66f6a4.pth?download=true) |
| RoI Transformer | 35.7 | [roi_trans_r50_fpn_1x_star_le90](configs/roi_trans/roi_trans_r50_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/roi_trans_r50_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/roi_trans_r50_fpn_1x_star_le90-e42f64d6.pth?download=true) |
| LSKNet-T | 34.7 | [lsk_t_fpn_1x_star_le90](configs/lsknet/lsk_t_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/lsk_t_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/lsk_t_fpn_1x_star_le90-19635614.pth?download=true) |
| LSKNet-S | 37.8 | [lsk_s_fpn_1x_star_le90](configs/lsknet/lsk_s_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/lsk_s_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/lsk_s_fpn_1x_star_le90-b77cdbc2.pth?download=true) |
| PKINet-S | 32.8 | [pkinet_s_fpn_1x_star_le90](configs/pkinet/pkinet_s_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/pkinet_s_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/pkinet_s_fpn_1x_star_le90-e1459201.pth?download=true) |
| ReDet | 39.1 | [redet_re50_refpn_1x_star_le90](configs/redet/redet_re50_refpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/redet_re50_refpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/redet_re50_refpn_1x_star_le90-d163f450.pth?download=true) | [ReResNet50](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/re_resnet50_c8_batch256-25b16846.pth?download=true) |
| Oriented RCNN | 40.7 | [oriented_rcnn_swin-l_fpn_1x_star_le90](configs/oriented_rcnn/oriented_rcnn_swin-l_fpn_1x_star_le90.py) | [log](https://huggingface.co/yangxue/STAR-MMRotate/raw/main/oriented_rcnn_swin-l_fpn_1x_star_le90.log) \| [ckpt](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/oriented_rcnn_swin-l_fpn_1x_star_le90-fe6f9e2d.pth?download=true) | [Swin-L](https://huggingface.co/yangxue/STAR-MMRotate/resolve/main/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth?download=true) |

## 🖊️ Citation

If you find this work helpful for your research, please consider giving this repo a star ⭐ and citing our paper:

```bibtex
@article{li2024star,
  title={STAR: A First-Ever Dataset and A Large-Scale Benchmark for Scene Graph Generation in Large-Size Satellite Imagery},
  author={Li, Yansheng and Wang, Linlin and Wang, Tingzhu and Yang, Xue and Luo, Junwei and Wang, Qi and Deng, Youming and Wang, Wenbin and Sun, Xian and Li, Haifeng and Dang, Bo and Zhang, Yongjun and Yi, Yu and Yan, Junchi},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```

## 📃 License

This project is released under the [Apache license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.
