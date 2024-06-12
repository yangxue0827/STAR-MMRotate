# Scene Graph Generation in Large-Size VHR Satellite Imagery: A Large-Scale Dataset and A Context-Aware Approach

The official implementation of the oriented object detection part of the paper "[Scene Graph Generation in Large-Size VHR Satellite Imagery: A Large-Scale Dataset and A Context-Aware Approach](https://arxiv.org/abs/)".

## ⭐️ Highlights

**TL;DR:** We propose RSG, the first large-scale dataset for scene graph generation in large-size VHR SAI. Containing more than `210,000` objects and over `400,000` triplets across `1,273` complex scenarios globally.

<p align="center">
<img src="demo/rsg.jpg" alt="scatter" width="98%"/> 
</p>

## 📌 Abstract

Scene graph generation (SGG) in satellite imagery (SAI) benefits promoting intelligent understanding of geospatial scenarios from perception to cognition. In SAI, objects exhibit great variations in scales and aspect ratios, and there exist rich relationships between objects (even between spatially disjoint objects), which makes it necessary to holistically conduct SGG in large-size very-high-resolution (VHR) SAI. However, the lack of SGG datasets with large-size VHR SAI has constrained the advancement of SGG in SAI. Due to the complexity of large-size VHR SAI, mining triplets $<$subject, relationship, object$>$ in large-size VHR SAI heavily relies on long-range contextual reasoning. Consequently, SGG models designed for small-size natural imagery are not directly applicable to large-size VHR SAI. To address the scarcity of datasets, this paper constructs a large-scale dataset for SGG in large-size VHR SAI with image sizes ranging from 512 × 768 to 27,860 × 31,096 pixels, named RSG, encompassing over 210,000 objects and more than 400,000 triplets. To realize SGG in large-size VHR SAI, we propose a context-aware cascade cognition (CAC) framework to understand SAI at three levels: object detection (OBD), pair pruning and relationship prediction. As a fundamental prerequisite for SGG in large-size SAI, a holistic multi-class object detection network (HOD-Net) that can flexibly integrate multi-scale contexts is proposed. With the consideration that there exist a huge amount of object pairs in large-size SAI but only a minority of object pairs contain meaningful relationships, we design a pair proposal generation (PPG) network via adversarial reconstruction to select high-value pairs. Furthermore, a relationship prediction network with context-aware messaging (RPCM) is proposed to predict the relationship types of these pairs. To promote the development of SGG in large-size VHR SAI, this paper releases a SAI-oriented SGG toolkit with 3 OBD methods and 5 SGG methods, and develops a benchmark based on RSG where our RPCM outperforms the SOTA methods with a large margin of 3.65\%/5.17\%/3.80\% at HMR@1500 on PredCls/SGCls/SGDet. The RSG dataset and SAI-oriented toolkit will be made publicly available at https://linlin-dev.github.io/project/RSG.html.

## 🛠️ Usage

For instructions on installation, pretrained models, training and evaluation, please refer to [mmrotate](README_en.md).

## 🚀 Released Models

### Oriented Object Detection

|  Detector  | mAP | Configs | Download |
| :--------: |:---:|:-------:|:--------:|
|  KLD  |  25.0  |   [rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc](configs/kld/rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/rsg-mmrotate/raw/main/rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/rsg-mmrotate/resolve/main/rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc-343a0b83.pth?download=true) |
|  GWD  |  25.3  |   [rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc](configs/gwd/rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/rsg-mmrotate/raw/main/rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/rsg-mmrotate/resolve/main/rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc-566d2398.pth?download=true) |
| KFIoU |  25.5  |   [rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc](configs/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/rsg-mmrotate/raw/main/rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/rsg-mmrotate/resolve/main/rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc-198081a6.pth?download=true) |
| FCOS  |  28.1  |   [rotated_fcos_r50_fpn_1x_rsg_le90](configs/rotated_fcos/rotated_fcos_r50_fpn_1x_rsg_le90.py)  |  [log](https://huggingface.co/yangxue/rsg-mmrotate/raw/main/rotated_fcos_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/rsg-mmrotate/resolve/main/rotated_fcos_r50_fpn_1x_rsg_le90-a579fbf7.pth?download=true) | 
|  KLD  |  25.0  |   [](configs//.py)  |  [log](https://huggingface.co/yangxue/rsg-mmrotate/raw/main/.log) \| [ckpt]() |

## 🖊️ Citation

If you find this work helpful for your research, please consider giving this repo a star ⭐ and citing our paper:

```bibtex
@article{li2024scene,
  title={Scene Graph Generation in Large-Size VHR Satellite Imagery: A Large-Scale Dataset and A Context-Aware Approach},
  author={L1, Yansheng and Wang, Linlin and Wang, Tingzhu and Wang, Qi and Sun, Xian and Yang, Xue and Wang, Wenbin and Luo, Junwei and Deng, Youming and Li, Haifeng and Dang, Bo and Zhang, Yongjun and Yan Junchi},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```

## 📃 License

This project is released under the [Apache license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.