

 
 # <p align=center> OneForecast: A Universal Framework for Global and Regional Weather Forecasting</p>

 <div align="center">
 
[![ArXiv](https://img.shields.io/badge/OneForecast-ArXiv-red.svg)](https://arxiv.org/abs/2502.00338)
[![Paper](https://img.shields.io/badge/OneRestore-Paper-yellow.svg)](https://openreview.net/forum?id=9xGSeVolcN)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/YuanGao-YG/OneForecast/tree/main)

</div>
<div align=center>
<img src="img/fig_main.jpg" width="1080">
</div>

---
>**OneForecast: A Universal Framework for Global and Regional Weather Forecasting**<br>  [Yuan Gao](https://scholar.google.com.hk/citations?hl=zh-CN&user=4JpRnU4AAAAJ&view_op=list_works&sortby=pubdate)<sup>† </sup>, [Hao Wu](https://easylearningscores.github.io/)<sup>† </sup>, [Ruiqi Shu](https://scholar.google.com.hk/citations?user=WKBB3r0AAAAJ&hl=zh-CN&oi=sra)<sup>† </sup>, [Huanshuo Dong](https://scholar.google.com.hk/citations?hl=zh-CN&user=VdGW_n8AAAAJ&view_op=list_works&sortby=pubdate), [Fan Xu](https://scholar.google.com.hk/citations?hl=zh-CN&user=qfMSkBgAAAAJ&view_op=list_works&sortby=pubdate), [Rui Ray Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=hM32GugAAAAJ&view_op=list_works&sortby=pubdate), [Yibo Yan](https://scholar.google.com.hk/citations?hl=zh-CN&user=26yPSEcAAAAJ&view_op=list_works&sortby=pubdate), [Qingsong Wen](https://sites.google.com/site/qingsongwen8/), [Xuming Hu](https://xuminghu.github.io/), [Kun Wang](https://scholar.google.com/citations?user=UnyqjWQAAAAJ&hl=en&oi=sra), [Jiahao Wu](https://scholar.google.com/citations?user=GuQ10J4AAAAJ&hl=zh-CN), [Qing Li](https://www4.comp.polyu.edu.hk/~csqli/), [Hui Xiong](https://scholar.google.com.hk/citations?hl=zh-CN&user=cVDF1tkAAAAJ&view_op=list_works&sortby=pubdate), [Xiaomeng Huang](http://faculty.dess.tsinghua.edu.cn/huangxiaomeng/en/index.htm)<sup>* </sup> <br>
(† Equal contribution, * Corresponding Author)<br>


> **Abstract:** *Accurate weather forecasts are important for disaster prevention, agricultural planning, etc. Traditional numerical weather prediction (NWP) methods offer physically interpretable high-accuracy predictions but are computationally expensive and fail to fully leverage rapidly growing historical data. In recent years, deep learning models have made significant progress in weather forecasting, but challenges remain, such as balancing global and regional high-resolution forecasts, excessive smoothing in extreme event predictions, and insufficient dynamic system modeling. To address these issues, this paper proposes a global-regional nested weather forecasting framework (OneForecast) based on graph neural networks. By combining a dynamic system perspective with multi-grid theory, we construct a multi-scale graph structure and densify the target region to capture local high-frequency features. We introduce an adaptive messaging mechanism, using dynamic gating units to deeply integrate node and edge features for more accurate extreme event forecasting. For high-resolution regional forecasts, we propose a neural nested grid method to mitigate boundary information loss. Experimental results show that OneForecast performs excellently across global to regional scales and short-term to long-term forecasts, especially in extreme event predictions. Codes link: \url{https://github.com/YuanGao-YG/OneForecast}.*
---

## News 🚀
* **2025.06.03**: Training codes are released.
* **2025.05.01**: OneForecast is accepted by [ICML 2025](https://icml.cc/).
* **2025.02.15**: Inference codes and pre-trained weights are released.
* **2025.02.03**: Codes for models are released.
* **2025.02.01**: Paper is released on [ArXiv](http://arxiv.org/abs/2502.00338).

## Notes

The intact project is avilable on [Hugging Face](https://huggingface.co/YuanGao-YG/OneForecast/tree/main), you can find the pretrained models, test data on Hugging Face and put them in the same location.


## Quick Start

### Installation

- cuda 11.8

```
# git clone this repository
git clone https://github.com/YuanGao-YG/OneForecast.git
cd OneForecast

# create new anaconda env
conda env create -f environment.yml
conda activate oneforecast
```


### Inference

1. Global Forecasts Inference

(1) Preparing the test data as follows:

```
./data/
|--global
|  |--test
|  |  |--2020.h5
|  |--mean.npy
|  |--std.npy
```

(2) Inference with 1-step supervised pretrained ckpt:
```
sh inference.sh
```

(3) Inference with finetuned pretrained ckpt:
```
sh inference_finetune.sh
```

2. Regional Forecasts Inference

(1) Preparing the test data as follows:

```
./data/
|--global
|  |--test
|  |  |--2020.h5
|  |--mean.npy
|  |--std.npy
|--regional
|  |--test
|  |  |--2020.h5
```

(2) Inference with 1-step supervised pretrained ckpt:

```
sh inference_nng.sh
```
   
## Training

### Global Forecasts

**1. Prepare Data**

Preparing the train, valid, and test data as follows:

```
./data/
|--global
|  |--train
|  |  |--1959.h5
|  |  |--1960.h5
|  |  |--.......
|  |  |--2016.h5
|  |  |--2017.h5
|  |--valid
|  |  |--2017.h5
|  |  |--2018.h5
|  |--test
|  |  |--2020.h5
|  |--mean.npy
|  |--std.npy
```

Each h5 file includes a key named 'fields' with the shape [T, C, H, W] (T=1460/1464, C=69, H=121, W=240)


**2. Model Training with 1-step Supervision**

- **Multi-node Multi-GPU Training**

```
sh train.sh
```

**3. Model Training with Multi-step Supervision Finetune**

- **Multi-node Multi-GPU Training**

(1) Modify `./train_finetune.sh` file and `./config/Model.yaml` file.

For instance, if you intent to finetune ckpt from 1-step ckpt (the start training time is 20250603-190101) with 2-step finetune for 10 eppchs, you can set `run_num='20250603-190101'`, `multi_steps_finetune=2`, `finetune_max_epochs=10`, `lr: 1E-6`. Please note that using a small learning rate (lr) to finetune model may contribute to convergence, you can adjust it according to your total batch size.

If you intent to finetune ckpt from 2-step ckpt (the start training time is 20250603-190101) with 3-step finetune for 10 eppchs, you can set `run_num='20250603-190101'`, `multi_steps_finetune=3`, `finetune_max_epochs=10`, `lr: 1E-6`.

(2) Run the following script:

```
sh train_finetune.sh
```

### Regional Forecasts

**1. Prepare Data**

Preparing the train, valid, and test data as follows:

```
./data/
|--global
|  |--train
|  |  |--1959.h5
|  |  |--1960.h5
|  |  |--.......
|  |  |--2016.h5
|  |  |--2017.h5
|  |--valid
|  |  |--2017.h5
|  |  |--2018.h5
|  |--test
|  |  |--2020.h5
|  |--mean.npy
|  |--std.npy
|--regional
|  |--train
|  |  |--1959.h5
|  |  |--1960.h5
|  |  |--.......
|  |  |--2016.h5
|  |  |--2017.h5
|  |--valid
|  |  |--2017.h5
|  |  |--2018.h5
|  |--test
|  |  |--2020.h5
```

Each h5 file includes a key named 'fields' with the shape [T, C, H, W] (T=1460/1464, C=69, H=721, W=1440)


**2. Model Training**

- **Multi-node Multi-GPU Training**

(1) Modify `./train_nng.sh` file and `./config/Model_nng.yaml` file.

Before training the regional model, a pretrained global ckpt is necessary. For instance, if you intent to train the regional model using global model (the start training time is 20250603-190101) as forecing, you can set `run_num='20250603-190101'`, `multi_steps_finetune=1`, `finetune_max_epochs=200`, `lr: 1E-3`. You can also adjust the learning rate (lr) according to your total batch size.

(2) Run the following script:

```
sh train_nng.sh
```

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
<img src="img/fig_vis.jpg" width="1080">
</div>

### Regional Forecasts

</div>
<div align=center>
<img src="img/fig_region.jpg" width="1080">
</div>

### Extreme Event Forecasts (Typhoon)

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

#### If you have any questions, please contact [yuangao24@mails.tsinghua.edu.cn](mailto:yuangao24@mails.tsinghua.edu.cn), [wuhao2022@mail.ustc.edu.cn](mailto:wuhao2022@mail.ustc.edu.cn), [srq24@mails.tsinghua.edu.cn](mailto:srq24@mails.tsinghua.edu.cn)

**Acknowledgment:** This code is based on the [NVIDIA modulus](https://github.com/NVIDIA/modulus).

