# LargeBatchCTR

LargeBatchCTR aims to train CTR prediction networks with large batch (>128k). The framework is based on [DeepCTR](https://github.com/zhengzangw/LargeBatchCTR). You can run the code on a V100 GPU to feel the fast training speed.

## Get Started

First, download dataset to the data folder. Use `data_utils.py` to preprocess the data for training.

```sh
python data_utils --dataset criteo_kaggle --split rand
```

Then, use `train.py` to train the network.

```sh
python train.py
```

For large batch training, do as follows:

```sh
python train.py
```

## Dataset List

- [Criteo Kaggle](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset): put `train.txt` in `data/criteo_kaggle/`
- [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction): put `train` in `data/avazu`
- Criteo Terabyte
- [Taobao (Alibaba)](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)

## Model List

- DeepFM
