### 介绍
本仓库为跟随[d2l](https://zh.d2l.ai/)的学习记录,包括所有的原始代码和部分数据集

### 数据集记录
使用了如下数据集:
1. Fashion-MNIST
   - 来源:[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
   - 通过`torchvision.datasets.FashionMNIST`自动下载
   - 提供`fmnist_load.py`加载
2. House Prices — Advanced Regression Techniques
   - 来源:[Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
   - 在比赛页面data部分下载
   - 相关代码和数据集也在本项目目录`house_price/`