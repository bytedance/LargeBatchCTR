import argparse
from math import sqrt

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from deepctr.feature_column import DenseFeat, SparseFeat, get_feature_names
from deepctr.models import DeepFM

from .utils import create_logdir, print_curtime, tf_allow_growth
from .data_utils import load_data, load_feature_name, auc


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int)  # 1234, 1235, 1236
    parser.add_argument("--dataset", default="criteo_kaggle",
                        choices=["criteo_kaggle", "criteo_terabyte", "avazu", "alibaba"], type=str)
    parser.add_argument("--model", choices=["DeepFM"], default="DeepFM")

    # Debug
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--log_freq", default=0, type=int)

    # Hyperparemters
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--sparse_embed_dim", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--l2", type=float, default=1e-5)

    # HPs
    parser.add_argument("--bs", default=1024, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_embed", default=1e-4, type=float)
    parser.add_argument("--l2_embed", type=float, default=1e-5)

    # HPs
    parser.add_argument("--clip", type=float, default=0)
    parser.add_argument("--warmup", type=float, default=0)
    parser.add_argument("--init_stddev", type=float, default=1e-2)

    args = parser.parse_args()
    return args


def get_feature_column(data, sparse_features, dense_features):
    sparse_feature_columns = [
        SparseFeat(
            feat,
            vocabulary_size=data[feat].max() + 1,
            embedding_dim=args.sparse_embed_dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=args.init_stddev, seed=2020)
        )
        for feat in sparse_features
    ]
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_features]
    fixlen_feature_columns = sparse_feature_columns + dense_feature_columns

    dnn_feature_columns = linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    return feature_names, dnn_feature_columns, linear_feature_columns


class CustomModel(tf.keras.Model):
    pass


if __name__ == "__main__":
    print_curtime("Program Start")
    args = parseargs()
    print(args)

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    tf_allow_growth()
    log_dir = create_logdir(args=args)

    sparse_features, dense_features, target = load_feature_name(args.dataset)
    train, test = load_data(args.dataset)

    # Define feature
    feature_names, dnn_feature_columns, linear_feature_columns = get_feature_column(
        train, sparse_features, dense_features
    )
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    y_train = train[target].values
    y_test = test[target].values.astype(np.float64)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # =====
        # Model
        # =====
        model_args = dict(
            dnn_hidden_units=(400, 400, 400),
            dnn_dropout=args.dropout,
            l2_reg_linear=0,
            l2_reg_embedding=0,
            custom_model=CustomModel,
        )
        if args.model == "DeepFM":
            model_class = DeepFM
        model = model_class(linear_feature_columns,
                            dnn_feature_columns, **model_args)

        # =====
        # Optimizer
        # =====
        layers = [
            [
                x
                for x in model.layers
                if "sparse_emb_" in x.name or "linear0sparse_emb_" in x.name
            ],
            [
                x
                for x in model.layers
                if "sparse_emb_" not in x.name and "linear0sparse_emb_" not in x.name
            ],
        ]
        num_step_per_epoch = int(len(y_train) / args.bs)

        if args.warmup > 0:
            learning_rate_fn = args.lr_embed
            lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                1e-8, int(args.warmup * num_step_per_epoch), args.lr, power=1
            )
        else:
            learning_rate_fn = args.lr_embed
            lr_fn = args.lr

        if args.opt == "adam":
            optimizers = [
                tf.keras.optimizers.Adam(learning_rate=learning_rate_fn),
                tf.keras.optimizers.Adam(learning_rate=lr_fn),
            ]
        elif args.opt == "adagrad":
            optimizers = [
                tf.keras.optimizers.Adagrad(learning_rate=learning_rate_fn),
                tf.keras.optimizers.Adam(learning_rate=lr_fn),
            ]
        elif args.opt == "ftrl":
            optimizers = [
                tf.keras.optimizers.Ftrl(learning_rate=learning_rate_fn),
                tf.keras.optimizers.Adam(learning_rate=lr_fn),
            ]
        else:
            raise NotImplementedError

        optimizers_and_layers = list(zip(optimizers, layers))
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    # =====
    # Training
    # =====
    model.compile(
        optimizer,
        tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        ),
        metrics=["binary_crossentropy", auc],
        run_eagerly=args.eager,
    )

    print_curtime("Start Training")
    model.fit(
        train_model_input,
        y_train,
        batch_size=args.bs,
        epochs=args.epoch,
        verbose=1,
        validation_data=(test_model_input, y_test),
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir, write_graph=False, update_freq=args.log_freq
            )
        ],
    )
    print_curtime("Program Ended")
