

 # <p align=center> OneForecast: A Universal Framework for Global and Regional Weather Forecasting</p>

</div>
<div align=center>
<img src="img/fig_main.jpg" width="1080">
</div>

---
>**OneForecast: A Universal Framework for Global and Regional Weather Forecasting**<br>  [Yuan Gao](https://scholar.google.com.hk/citations?hl=zh-CN&user=4JpRnU4AAAAJ&view_op=list_works&sortby=pubdate)<sup>† </sup>, [Hao Wu](https://easylearningscores.github.io/)<sup>† </sup>, [Ruiqi Shu](https://scholar.google.com.hk/citations?user=WKBB3r0AAAAJ&hl=zh-CN&oi=sra)<sup>† </sup>, Huanshuo Dong, [Fan Xu](https://scholar.google.com.hk/citations?hl=zh-CN&user=qfMSkBgAAAAJ&view_op=list_works&sortby=pubdate), Rui Chen, [Yibo Yan](https://scholar.google.com.hk/citations?hl=zh-CN&user=26yPSEcAAAAJ&view_op=list_works&sortby=pubdate), [Qingsong Wen](https://sites.google.com/site/qingsongwen8/), [Xuming Hu](https://xuminghu.github.io/), [Kun Wang](https://scholar.google.com/citations?user=UnyqjWQAAAAJ&hl=en&oi=sra), [Jiahao Wu](https://scholar.google.com/citations?user=GuQ10J4AAAAJ&hl=zh-CN), [Qing Li](https://www4.comp.polyu.edu.hk/~csqli/), [Hui Xiong](https://scholar.google.com.hk/citations?hl=zh-CN&user=cVDF1tkAAAAJ&view_op=list_works&sortby=pubdate), [Xiaomeng Huang](http://faculty.dess.tsinghua.edu.cn/huangxiaomeng/en/index.htm)<sup>* </sup> <br>
(† Equal contribution, * Corresponding Author)<br>


> **Abstract:** *Accurate weather forecasts are important for disaster prevention, agricultural planning, and water resource management. Traditional numerical weather prediction (NWP) methods offer physically interpretable high-accuracy predictions but are computationally expensive and fail to fully leverage rapidly growing historical data. In recent years, deep learning methods have made significant progress in weather forecasting, but challenges remain, such as balancing global and regional high-resolution forecasts, excessive smoothing in extreme event predictions, and insufficient dynamic system modeling. To address these issues, this paper proposes a global-regional nested weather forecasting framework based on graph neural networks (GNNs). By combining a dynamic system perspective with multi-grid theory, we construct a multi-scale graph structure and densify the target region to capture local high-frequency features. We introduce an adaptive information propagation mechanism, using dynamic gating units to deeply integrate node and edge features for more accurate extreme event forecasting. For high-resolution regional forecasts, we propose a neural nested grid method to mitigate boundary information loss. Experimental results show that the proposed method performs excellently across global to regional scales and short-term to long-term forecasts, especially in extreme event predictions (e.g., typhoons), significantly improving forecast accuracy.*
---

## News 🚀
* **2025.02.15**: Inference codes and pre-trained weights released.
* **2025.02.03**: Codes for models are released.
* **2025.02.01**: Paper is released on [ArXiv](http://arxiv.org/abs/2502.00338).

## Quick Start

### Install

continue update

### Pretrained Models

| Model | OneDrive | Description
| :--- | :--- | :----------
|ckpt_OneForecast.tar | [model](https://1drv.ms/u/s!AvMfp6yKSwiXbcQcHdcPJvSGG_Q?e=UBrb1o) | Pre-trained weights of OneForecast.


### Inference

continue update

## Training
### Global Forecasts
**1. Prepare Data**

Preparing the train, valid, and test data as follows:

```
./data/global/
|--train
|  |--1959.h5
|  |--1960.h5
|  |--......
|  |--2016.h5
|  |--2017.h5
|--valid
|  |--2018.h5
|  |--2019.h5
|--test
|  |--2020.h5
```

**2. Model Training**


- **Single GPU Training**
- **Single-node Multi-GPU Training**
- **Multi-node Multi-GPU Training**

### Regional Forecasts
**1. Prepare Data**

Preparing the train, valid, and test data as follows:

```
./data/regional/
|--train
|  |--1959.h5
|  |--1960.h5
|  |--......
|  |--2016.h5
|  |--2017.h5
|--valid
|  |--2018.h5
|  |--2019.h5
|--test
|  |--2020.h5
```

**2. Model Training**
- **Single GPU Training**
- **Single-node Multi-GPU Training**
- **Multi-node Multi-GPU Training**


## Performance
### Global Forecasts

</div>
<div align=center>
<img src="img/tb_metric.jpg" width="1080">
</div>

</div>
<div align=center>
<img src="img/fig_metric.jpg" width="1080">
</div>

</div>
<div align=center>
<img src="img/fig_visual.jpg" width="1080">
</div>

### Regional Forecasts

</div>
<div align=center>
<img src="img/fig_region.jpg" width="1080">
</div>

### Extreme Event Forecasts

</div>
<div align=center>
<img src="img/fig_typhoon.jpg" width="1080">
</div>


## Citation

```
@article{gao2025oneforecast,
  title={OneForecast: A Universal Framework for Global and Regional Weather Forecasting},
  author={Gao, Yuan and Wu, Hao and Shu, Ruiqi and Dong, Huanshuo and Xu, Fan and Chen, Rui and Yan, Yibo and Wen, Qingsong and Hu, Xuming and Wang, Kun and others},
  journal={arXiv preprint arXiv:2502.00338},
  year={2025}
}
```

#### If you have any questions, please contact [yuangao24@mails.tsinghua.edu.cn](mailto:yuangao24@mails.tsinghua.edu.cn).

**Acknowledgment:** This code is based on the [NVIDIA modulus](https://github.com/NVIDIA/modulus).

