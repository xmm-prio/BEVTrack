<h1 align="center">BEVTrack: A Simple Baseline for 3D single objet Tracking in Bird's-Eye-View</h1>
<p align="center">
<a href="https://arxiv.org/abs/2309.02185"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a>
<h5 align="center"><em>Yuxiang Yang, Yingqi Deng, Jing Zhang, Jiahao Nie, Zheng-Jun Zha</em></h5>
<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#results">Results</a> |
  <a href="#statement">Statement</a>
</p>
<figure>
<p align="center">
 <img src="Figs/bevtrack.png">
</figure>

## Introduction
<p align="justify">3D single object tracking (SOT) in point clouds is still a challenging problem due to appearance variation, distractors, and high sparsity of point clouds. Notably, in autonomous driving scenarios, the target object typically maintains spatial adjacency across consecutive frames, predominantly moving horizontally. This spatial continuity offers valuable prior knowledge for target localization. However, existing trackers, which often employ point-wise representations, struggle to efficiently utilize this knowledge owing to the irregular format of such representations. Consequently, they require elaborate designs and solving multiple subtasks to establish spatial correspondence. In this paper, we introduce BEVTrack, a simple yet strong baseline framework for 3D SOT. After converting consecutive point clouds into the common Bird's-Eye-View representation, BEVTrack inherently encodes spatial proximity and adeptly captures motion cues for tracking via a simple element-wise operation and convolutional layers. Additionally, to better deal with objects having diverse sizes and moving patterns, BEVTrack directly learns the underlying motion distribution rather than making a fixed Laplacian or Gaussian assumption as in previous works. Without bells and whistles, BEVTrack achieves state-of-the-art performance on KITTI and NuScenes datasets while maintaining a high inference speed of 122 FPS. 

## Results
### Comparison with State-of-the-art Methods
<figure>
<p align="center">
 <img src="Figs/kitti.png">
</figure>

<figure>
<p align="center">
 <img src="Figs/nuscenes.png">
</figure>

## Statement

<p align="justify">If you are interested in our work, please consider citing the following:</p>

```
@misc{2309.02185,
Author = {Yuxiang Yang, Yingqi Deng, Jing Zhang, Jiahao Nie, Zheng-Jun Zha},
Title = {BEVTrack: A Simple Baseline for 3D single object Tracking in Bird's-Eye-View},
Year = {2023},
Eprint = {arXiv:2309.02185},
}
```
<p align="justify">This project is under MIT licence.</p>

## Relevant Projects

[1] <strong>GLT-T: Global-Local Transformer Voting for 3D Single Object Tracking in Point Clouds, AAAI, 2023</strong> | [Paper](https://arxiv.org/abs/2211.10927) | [Github](https://github.com/haooozi/GLT-T) <br><em>&ensp; &ensp; &ensp;Jiahao Nie<sup>&#8727;</sup>, Zhiwei He<sup>&#8727;</sup>, Yuxiang Yang, Mingyu Gao, Jing Zhang</em>

[2] <strong>OSP2B: One-Stage Point-to-Box Network for 3D Siamese Tracking, IJCAI, 2023</strong> | [Paper](https://arxiv.org/abs/2304.11584) | [Github](https://github.com/haooozi/OSP2B) <br><em>&ensp; &ensp; &ensp;Jiahao Nie<sup>&#8727;</sup>, Zhiwei He<sup>&#8727;</sup>, Yuxiang Yang, Zhengyi Bao, Mingyu Gao, Jing Zhang</em>
