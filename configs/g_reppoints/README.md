# G-Rep

> G-Rep: Gaussian Representation for Arbitrary-Oriented Object Detection.

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://private-user-images.githubusercontent.com/29257168/340519456-db4647a8-74bf-4b2f-ae61-d4fa16654bc0.png" width="800"/>
</div>

Typical representations for arbitrary-oriented object detection tasks include oriented bounding box (OBB), quadrilateral bounding box (QBB), and point set (PointSet). Each representation encounters problems that correspond to its characteristics, such as the boundary discontinuity, square-like problem, representation ambiguity, and isolated points, which lead to inaccurate detection. Although many effective strategies have been proposed for various representations, there is still no unified solution. Current detection methods based on Gaussian modeling have demonstrated the possibility of breaking this dilemma; however, they remain limited to OBB. To go further, in this paper, we propose a unified Gaussian representation called G-Rep to construct Gaussian distributions for OBB, QBB, and PointSet, which achieves a unified solution to various representations and problems. Specifically, PointSet or QBB-based object representations are converted into Gaussian distributions, and their parameters are optimized using the maximum likelihood estimation algorithm. Then, three optional Gaussian metrics are explored to optimize the regression loss of the detector because of their excellent parameter optimization mechanisms. Furthermore, we also use Gaussian metrics for sampling to align label assignment and regression loss. Experimental results on several public available datasets, such as DOTA, HRSC2016, UCAS-AOD, and ICDAR2015, show the excellent performance of the proposed method for arbitrary-oriented object detection.

## Results and models

DOTA1.0


|         Backbone         |  mAP  | Angle | lr schd | Mem (GB) | Inf Time (fps) | Aug | Batch Size |                                               Configs                                               |                                                                                                                                                                    Download                                                                                                                                                                    |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :--------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 59.44 |  oc  |   1x   |   3.45   |      15.6      |  -  |     2     | [rotated_reppoints_r50_fpn_1x_dota_oc](../rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc/rotated_reppoints_r50_fpn_1x_dota_oc-d38ce217.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc/rotated_reppoints_r50_fpn_1x_dota_oc_20220205_145010.log.json) |
| ResNet50 (1024,1024,200) | 69.49 | le135 |   1x   |   4.05   |      8.6      |  -  |     2     |             [g_reppoints_r50_fpn_1x_dota_le135](./g_reppoints_r50_fpn_1x_dota_le135.py)             |             [model](https://download.openmmlab.com/mmrotate/v0.1.0/g_reppoints/g_reppoints_r50_fpn_1x_dota_le135/g_reppoints_r50_fpn_1x_dota_le135-b840eed7.pth) \| [log](https://download.openmmlab.com/mmrotate/v0.1.0/g_reppoints/g_reppoints_r50_fpn_1x_dota_le135/g_reppoints_r50_fpn_1x_dota_le135_20220202_233631.log.json)             |

## Citation

Coming soon!
