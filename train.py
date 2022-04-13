import argparse
from math import sqrt

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import log_loss, roc_auc_score

from clip import cow_clip
from data_utils import load_data, load_feature_name
from deepctr.feature_column import DenseFeat, SparseFeat, get_feature_names
from deepctr.models import DCN, WDL, DCNMix, DeepFM
from deepctr.models.widefm import wideFM
from utils import auc, create_logdir, print_curtime, tf_allow_growth, num_params


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1235, type=int)  # 1234, 1235, 1236
    parser.add_argument("--dataset", default="criteo_kaggle",
                        choices=["criteo_kaggle", "avazu"], type=str)
    parser.add_argument("--split", default="rand",
                        choices=["rand", "seq", "highfreq"])
    parser.add_argument(
        "--model", choices=["LR", "FM", "WD", "DeepFM", "xDeepFM", "DCN", "DCNv2"], default="DeepFM")

    # Debug
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--profile", action="store_true")

    # Hyperparemters
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--sparse_embed_dim", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0)

    # HPs
    parser.add_argument("--bs", default=1024, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_embed", default=1e-4, type=float)
    parser.add_argument("--l2", type=float, default=1e-5)

    # HPs
    parser.add_argument("--opt", type=str, default="adam")
    parser.add_argument("--clip", type=float, default=0)
    parser.add_argument("--warmup", type=float, default=0)
    parser.add_argument("--init_stddev", type=float, default=1e-4)
    parser.add_argument("--bound", type=float, default=0)

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


def run_test(model, test_model_input, y_test):
    pred_ans = model.predict(test_model_input, batch_size=args.bs)
    pred_ans = pred_ans.astype(np.float64)
    return round(log_loss(y_test, pred_ans), 5), round(
        roc_auc_score(y_test, pred_ans), 5
    )


class CustomModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def train_step(self, data):
        ret = dict()

        # log setting
        self.cur_step.assign_add(1)

        def should_record(): return tf.equal(
            tf.math.floormod(self.cur_step, args.log_freq), 0
        )
        record_env = tf.summary.record_if(should_record)
        tf.summary.experimental.set_step(self.cur_step)

        # assist vars
        name_to_layer = {x.name: x for x in self.trainable_variables}
        uniq_ids, uniq_cnt = dict(), dict()
        for k, v in data[0].items():
            if k[0] != "I":
                y, _, count = tf.unique_with_counts(v)
                uniq_ids[k] = y
                uniq_cnt[k] = count

        # main
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # loss = self.compiled_loss(y, y_pred)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # clip
        name_to_gradient = {
            x.name: g for x, g in zip(self.trainable_variables, gradients)
        }
        embed_index = [
            i for i, x in enumerate(trainable_vars) if "embeddings" in x.name
        ]
        dense_index = [i for i in range(
            len(trainable_vars)) if i not in embed_index]
        embed_vars = [trainable_vars[i] for i in embed_index]
        dense_vars = [trainable_vars[i] for i in dense_index]
        embed_gradients = [gradients[i] for i in embed_index]
        dense_gradients = [gradients[i] for i in dense_index]

        # CowClip
        if args.clip > 0:
            lower_bound = args.clip * sqrt(args.sparse_embed_dim) * args.bound
            embed_gradients_clipped = []
            for w, g in zip(embed_vars, embed_gradients):
                if 'linear' in w.name:
                    embed_gradients_clipped.append(g)
                    continue
                prefix = "sparse_emb_"
                col_name = w.name[
                    w.name.find(prefix) + len(prefix): w.name.find("/")
                ]

                g_clipped = cow_clip(w, g, ratio=args.clip,
                                         ids=uniq_ids[col_name], cnts=uniq_cnt[col_name], min_w=lower_bound)
                embed_gradients_clipped.append(g_clipped)

            embed_gradients = embed_gradients_clipped

        gradients = embed_gradients + dense_gradients

        # update
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # =====
        # Logging
        # =====
        embed_gradients = [gradients[i] for i in embed_index]
        dense_gradients = [gradients[i] for i in dense_index]
        with record_env:
            tf.summary.scalar(
                "lr/dense", self.optimizer.optimizer_specs[1]['optimizer']._decayed_lr('float32'))
            tf.summary.scalar("loss/loss", loss)
            tf.summary.scalar("global_norm/global",
                              tf.linalg.global_norm(gradients))
            tf.summary.scalar("global_norm/dense",
                              tf.linalg.global_norm(dense_gradients))
            tf.summary.scalar("global_norm/embed",
                              tf.linalg.global_norm(embed_vars))
            tf.summary.scalar("global_norm/var_dense",
                              tf.linalg.global_norm(dense_vars))
            tf.summary.scalar("global_norm/var_embed",
                              tf.linalg.global_norm(embed_gradients))

            if args.log:
                for i, (variable, gradient) in enumerate(
                    zip(trainable_vars, gradients)
                ):
                    name = variable.name
                    opt_index = 0 if i in embed_index else 1
                    m = self.optimizer.optimizer_specs[opt_index]["optimizer"].get_slot(
                        variable, "m"
                    )
                    v = self.optimizer.optimizer_specs[opt_index]["optimizer"].get_slot(
                        variable, "v"
                    )

                    layer_norm = tf.norm(variable)
                    grad_norm = tf.norm(gradient)
                    m_norm = tf.norm(m)
                    v_norm = tf.norm(v)

                    tf.summary.scalar("layer_norm/" + name, layer_norm)
                    tf.summary.scalar("grad_norm/" + name, grad_norm)
                    tf.summary.scalar("m_norm/" + name, m_norm)
                    tf.summary.scalar("v_norm/" + name, v_norm)

        self.compiled_metrics.update_state(y, y_pred)
        for m in self.metrics:
            ret[m.name] = m.result()
        return ret


if __name__ == "__main__":
    print_curtime("Program Start")
    args = parseargs()
    print(args)

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    tf_allow_growth()
    log_dir = create_logdir(args=args)

    sparse_features, dense_features, target = load_feature_name(args.dataset)
    train, test = load_data(args.dataset, split=args.split)

    # Define feature
    feature_names, dnn_feature_columns, linear_feature_columns = get_feature_column(
        train, sparse_features, dense_features
    )
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    y_train = train[target].values
    y_test = test[target].values.astype(np.float64)

    # =====
    # Model
    # =====
    model_args = dict(
        dnn_hidden_units=(400, 400, 400),
        dnn_dropout=args.dropout,
        l2_reg_linear=args.l2,
        l2_reg_embedding=args.l2,
        keras_model=CustomModel,
        seed=args.seed,
    )
    if args.model == "FM":
        model_class = wideFM
    elif args.model == "DeepFM":
        model_class = DeepFM
    elif args.model == "WD":
        model_class = WDL
    elif args.model == "DCN":
        model_class = DCN
        model_args['cross_num'] = 3
    elif args.model == "DCNv2":
        model_class = DCNMix
        model_args['cross_num'] = 3

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = model_class(linear_feature_columns,
                            dnn_feature_columns, **model_args)
        num_params(model)

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

    # tensorboard logger
    tbcb_args = dict(write_graph=False, update_freq=args.log_freq)
    if args.profile:
        tbcb_args['histogram_freq'] = 1
        tbcb_args['profile_batch'] = '80,100'
    cb = tf.keras.callbacks.TensorBoard(
        log_dir, **tbcb_args
    )

    print_curtime("Start Training")
    model.fit(
        train_model_input,
        y_train,
        batch_size=args.bs,
        epochs=args.epoch,
        verbose=1,
        validation_data=(test_model_input, y_test),
        callbacks=[cb],
    )

    # =====
    # Test
    # =====
    logloss, auc = run_test(model, test_model_input, y_test)
    print(f"[Test] LogLoss = {logloss}, AUC = {auc}")
    print_curtime("Program Ended")
