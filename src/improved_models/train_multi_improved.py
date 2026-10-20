r"""
train_multi_improved.py - improved MULTILAYER feedforward ANN (fully connected only).

Flatten -> [Dense -> BatchNorm -> ReLU -> Dropout] x N -> Dense(15, softmax)
Optional Resizing layer (like your v4) inside the model, so the saved model still
accepts raw [0,1] 64x64x3 images.

Examples:
  python train_multi_improved.py --tag m_full_a --train-source full
  python train_multi_improved.py --tag m_full_b --train-source full --size 64 --hidden "512,256" --dropout 0.5
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import (Input, Normalization, Resizing, Flatten, Dense,
                                     BatchNormalization, Activation, Dropout)

from common import (IMG_SHAPE, NUM_CLASSES, add_common_args, channel_stats,
                    load_train, load_val, run_experiment)

parser = add_common_args(argparse.ArgumentParser())
parser.add_argument("--size", type=int, default=32, choices=[32, 48, 64],
                    help="input resolution fed to the dense layers")
parser.add_argument("--hidden", type=str, default="512,256,128",
                    help="comma-separated hidden layer sizes")
parser.add_argument("--dropout", type=float, default=0.4)
parser.add_argument("--l2", type=float, default=1e-4)
parser.add_argument("--no-bn", action="store_true", help="disable BatchNormalization")
args = parser.parse_args()

tf.keras.utils.set_random_seed(args.seed)

X, y = load_train(args.train_source)
Xv, yv = load_val()
print("Train:", X.shape, X.dtype, " Val:", Xv.shape)

mean, var = channel_stats(X)
hidden = [int(h) for h in args.hidden.split(",")]

layers = [
    Input(shape=IMG_SHAPE),
    Normalization(axis=-1, mean=mean, variance=var, name="input_standardize"),
]
if args.size != 64:
    layers.append(Resizing(args.size, args.size, name="resize"))
layers.append(Flatten())
for i, units in enumerate(hidden, start=1):
    layers.append(Dense(units, kernel_regularizer=regularizers.l2(args.l2), name=f"hidden_{i}"))
    if not args.no_bn:
        layers.append(BatchNormalization(name=f"bn_{i}"))
    layers.append(Activation("relu", name=f"relu_{i}"))
    layers.append(Dropout(args.dropout, name=f"dropout_{i}"))
layers.append(Dense(NUM_CLASSES, activation="softmax", name="output"))
model = Sequential(layers)

run_experiment(
    model, X, y, Xv, yv, args,
    name=f"improved_multilayer_{args.tag}",
    architecture=f"Multilayer ANN (input {args.size}x{args.size}, hidden {hidden})",
    config={"train_source": args.train_source, "size": args.size, "hidden": hidden,
            "dropout": args.dropout, "l2": args.l2, "batchnorm": not args.no_bn,
            "batch": args.batch, "lr": args.lr, "class_weights": args.class_weights,
            "augment": not args.no_augment, "seed": args.seed},
)
