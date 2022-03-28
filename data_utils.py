import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


sparse_features = {
    "criteo_kaggle": ["C" + str(i) for i in range(1, 27)],
    "avazu": [
        "hour",
        "C1",
        "banner_pos",
        "site_id",
        "site_domain",
        "site_category",
        "app_id",
        "app_domain",
        "app_category",
        "device_id",
        "device_ip",
        "device_model",
        "device_type",
        "device_conn_type",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
    ]
}
dense_features = {
    "criteo_kaggle": ["I" + str(i) for i in range(1, 14)],
    "avazu": []
}
target = {
    "criteo_kaggle": ["label"],
    "avazu": ["click"],
}
data_path = {
    "criteo_kaggle": "data/criteo_kaggle",
    "avazu": "data/avazu",
}
split_rand_ratio = {
    "criteo_kaggle": 0.1,
    "avazu": 0.2
}


def load_data(dataset="criteo_kaggle", split="rand"):
    train_file = os.path.join(
        data_path[dataset], f"{dataset}_processed_{split}_train.feather")
    test_file = os.path.join(
        data_path[dataset], f"{dataset}_processed_{split}_test.feather")

    train_data = pd.read_feather(train_file)
    test_data = pd.read_feather(test_file)

    return train_data, test_data


def load_feature_name(dataset="criteo_kaggle"):
    return sparse_features[dataset], dense_features[dataset], target[dataset]


def preprocess(data, sparse_features, dense_features):
    data[sparse_features] = data[sparse_features].fillna(
        "-1",
    )
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    if len(dense_features) > 0:
        data[dense_features] = data[dense_features].fillna(
            0,
        )
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="criteo_kaggle",
                        choices=["criteo_kaggle", "criteo_terabyte", "avazu", "taobao"], type=str)
    parser.add_argument("--split", default=None,
                        choices=["seq", "rand"], type=str)
    args = parser.parse_args()

    o_file = os.path.join(data_path[args.dataset], f"{args.dataset}_processed")
    if args.dataset == "criteo_kaggle":
        i_file = os.path.join(data_path[args.dataset], "train.txt")
        data = pd.read_csv(i_file, delimiter="\t", header=None)
        data.columns = pd.read_csv(os.path.join(
            data_path[args.dataset], "criteo_sample.txt")).columns
    elif args.dataset == "avazu":
        i_file = os.path.join(data_path[args.dataset], "train")
        data = pd.read_csv(i_file, dtype={"id": str})

    sparse_features, dense_features, target = load_feature_name(args.dataset)
    data = preprocess(data, sparse_features, dense_features)

    if args.split is not None:
        if args.split == "seq":
            if args.dataset == "criteo_kaggle":
                test_ratio = 1/7
                train_size = int((1-test_ratio) * len(data))
                train = data.iloc[:train_size]
                test = data.iloc[train_size:]
            else:
                raise NotImplementedError
        else:
            data_index = np.arange(len(data))
            test_ratio = split_rand_ratio[args.dataset]
            train_index, test_index = train_test_split(
                data_index, test_size=test_ratio)
            train = data.iloc[train_index]
            test = data.iloc[test_index]
        train.reset_index(drop=True).to_feather(
            o_file + f"_{args.split}_train.feather")
        test.reset_index(drop=True).to_feather(
            o_file + f"_{args.split}_test.feather")
    else:
        data.to_feather(o_file + ".feather")
