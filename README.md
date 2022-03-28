# LargeBatchCTR

LargeBatchCTR aims to train CTR prediction networks with large batch (>128k). The framework is based on [DeepCTR](https://github.com/zhengzangw/LargeBatchCTR). You can run the code on a V100 GPU to feel the fast training speed.

## Get Started

First, download dataset to the data folder. Use `data_utils.py` to preprocess the data for training.

```sh
python data_utils --dataset criteo_kaggle --split rand
```

Then, use `train.py` to train the network.

```sh
python train.py --dataset avazu --model DeepFM --bs 1024 --l2 1e-05 --lr 1e-4 --lr_embed 1e-4
```

For large batch training, do as follows:

```sh
python train.py --dataset avazu --model DeepFM --bs 65536 --l2 64e-05 --lr 64e-4 --lr_embed 1e-4 --clip 1 --warmup 1 --init_stddev 0.01
```

## Dataset List

- [Criteo Kaggle](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset): put `train.txt` in `data/criteo_kaggle/`
- [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction): put `train` in `data/avazu`
- Criteo Terabyte (to be added)

## Model List

- FM
- DeepFM
- DCN
- DCNv2
