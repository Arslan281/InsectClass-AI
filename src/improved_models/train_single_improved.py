r"""
train_single_improved.py - improved SINGLE-LAYER ANN.

Architecture is unchanged in kind: Flatten -> Dense(15, softmax). NO hidden layer.
The only addition is a fixed (non-trainable) per-channel standardization step
built into the model, so the saved model still accepts raw [0,1] 64x64x3 images
exactly like your original models.

Example:
  python train_single_improved.py --tag s_full_aug --train-source full
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Input, Normalization, Flatten, Dense

from common import (IMG_SHAPE, NUM_CLASSES, add_common_args, channel_stats,
                    load_train, load_val, run_experiment)

parser = add_common_args(argparse.ArgumentParser())
parser.add_argument("--l2", type=float, default=1e-3)
args = parser.parse_args()

tf.keras.utils.set_random_seed(args.seed)

X, y = load_train(args.train_source)
Xv, yv = load_val()
print("Train:", X.shape, X.dtype, " Val:", Xv.shape)

mean, var = channel_stats(X)          # statistics from TRAINING images only
print("Channel mean:", mean, " variance:", var)

model = Sequential([
    Input(shape=IMG_SHAPE),
    Normalization(axis=-1, mean=mean, variance=var, name="input_standardize"),
    Flatten(),
    Dense(NUM_CLASSES, activation="softmax",
          kernel_regularizer=regularizers.l2(args.l2), name="output"),
])

run_experiment(
    model, X, y, Xv, yv, args,
    name=f"improved_single_layer_{args.tag}",
    architecture="Single-layer ANN (Flatten -> Dense(15, softmax))",
    config={"train_source": args.train_source, "batch": args.batch, "lr": args.lr,
            "l2": args.l2, "class_weights": args.class_weights,
            "augment": not args.no_augment, "seed": args.seed},
)
