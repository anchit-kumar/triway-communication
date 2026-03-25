import gc
import os
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")   # also supported way
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")  # helps fragmentation on many setups

import sys
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision, layers

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("set_memory_growth failed:", e)

# Mixed precision is fine, but turn it off if you still OOM
mixed_precision.set_global_policy("mixed_float16")

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# -----------------------
# Dataset functions
# -----------------------
def make_datasets(data_root, img_size, batch, seed, val_split=0.1):
    """
    Build train/val datasets ONCE with a FIXED split seed so every trial evaluates on the same set.
    label_mode="int" => sparse labels (int32).
    """
    split_seed = 12345  # keep eval split constant across trials

    train_raw = keras.preprocessing.image_dataset_from_directory(
        data_root,
        validation_split=val_split, subset="training",
        seed=split_seed, image_size=(img_size, img_size),
        batch_size=batch, label_mode="int"
    )

    val_raw = keras.preprocessing.image_dataset_from_directory(
        data_root,
        validation_split=val_split, subset="validation",
        seed=split_seed, image_size=(img_size, img_size),
        batch_size=batch, label_mode="int"
    )

    class_names = train_raw.class_names

    aug = keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomTranslation(0.05, 0.05),
    ], name="augment")

    def pp_train(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = aug(x, training=True)
        return x, y

    def pp_val(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    options = tf.data.Options()
    options.deterministic = True

    # These two can vary by TF version; guard them so it never errors.
    try:
        options.experimental_optimization.apply_default_optimizations = False
        options.experimental_optimization.map_parallelization = False
    except Exception:
        pass

    train_ds = train_raw.map(pp_train, num_parallel_calls=1).prefetch(tf.data.AUTOTUNE).with_options(options)
    val_ds   = val_raw.map(pp_val,   num_parallel_calls=1).prefetch(tf.data.AUTOTUNE).with_options(options)

    return train_ds, val_ds, class_names

def write_labels(labels_path, class_names):
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    with open(labels_path, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print("labels.txt written:", labels_path)

# -----------------------
# Model (scratch, DW + SE)
# -----------------------
class ASLClassifier:
    def __init__(self, img_size, num_classes, width_mult=1.0, dropout=0.30):
        self.img_size = img_size
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.dropout = dropout
        self.model = self._build_model()

    def _F(self, ch):
        return max(8, int(ch * self.width_mult))

    @staticmethod
    def _dw_relu_bn(x, stride=1):
        x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        return x

    def _se(self, x, r=4):
        ch = x.shape[-1]
        s = layers.GlobalAveragePooling2D()(x)
        s = layers.Reshape((1, 1, ch))(s)
        s = layers.Dense(max(ch // r, 8), activation="relu", use_bias=True)(s)
        s = layers.Dense(ch, activation="sigmoid", use_bias=True)(s)
        return layers.Multiply()([x, s])

    def _block(self, x, out_ch, stride=1, use_se=True):
        x = self._dw_relu_bn(x, stride)
        x = layers.Conv2D(self._F(out_ch), 1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        if use_se:
            x = self._se(x)
        return x

    def _build_model(self):
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))

        x = layers.Conv2D(self._F(32), 3, strides=2, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)

        x = self._block(x, 64, 1)
        x = self._block(x, 128, 2)
        x = self._block(x, 128, 1)

        x = self._block(x, 256, 2)
        x = self._block(x, 256, 1)
        x = self._block(x, 512, 2)

        for _ in range(5):
            x = self._block(x, 512, 1)

        x = self._block(x, 768, 2)
        x = self._block(x, 1024, 1)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax", dtype="float32", name="prob")(x)
        return keras.Model(inputs, outputs, name="asl_scratch_ds_cnn_se")

    def compile(self, lr=3e-3, wd=1e-4, label_smooth=0.05):
        num_classes = self.num_classes

        def sparse_cce_with_smoothing(y_true, y_pred):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_one  = tf.one_hot(y_true, depth=num_classes, dtype=y_pred.dtype)
            if label_smooth and label_smooth > 0.0:
                y_one = y_one * (1.0 - label_smooth) + label_smooth / tf.cast(num_classes, y_one.dtype)
            return tf.keras.losses.categorical_crossentropy(y_one, y_pred, from_logits=False)

        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd),
            loss=sparse_cce_with_smoothing,
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )

# -----------------------
# Training helper (one trial)
# -----------------------
def train_single_trial(train_ds, val_ds, trial_id, trial_dir, img_size, num_classes):
    width_mult = random.choice([1.0, 1.25])
    dropout = random.choice([0.25, 0.30])

    clf = ASLClassifier(img_size=img_size, num_classes=num_classes,
                        width_mult=width_mult, dropout=dropout)

    ckpt_path = os.path.join(trial_dir, f"trial{trial_id:02d}_best.weights.h5")

    total_epochs = 20
    warmup = 2

    def lr_fn(epoch):
        import math
        base = 3e-3
        min_lr = 3e-4
        if epoch < warmup:
            return (epoch + 1) * (base / warmup)
        t = (epoch - warmup) / max(1, total_epochs - warmup)
        return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))

    cbs = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, min_delta=0.002,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.LearningRateScheduler(lr_fn, verbose=1),
    ]
    # monitor="val_accuracy" is the standard metric key produced from metric name="accuracy". :contentReference[oaicite:1]{index=1}

    clf.compile(lr=3e-3, wd=1e-4, label_smooth=0.05)
    clf.model.fit(train_ds, validation_data=val_ds, epochs=total_epochs, callbacks=cbs, verbose=2)

    # Evaluate final (weights already restored by EarlyStopping if triggered)
    _, val_acc = clf.model.evaluate(val_ds, verbose=0)

    return {
        "val_acc": float(val_acc),
        "width_mult": float(width_mult),
        "dropout": float(dropout),
        "ckpt_path": ckpt_path,
    }

# -----------------------
# Main / CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to training data (subfolders = classes)")
    ap.add_argument("--out", default="./outputs", help="Output dir")
    ap.add_argument("--img", type=int, default=224, help="Input image size (NxN)")
    ap.add_argument("--batch", type=int, default=64, help="Batch Size")
    ap.add_argument("--target-acc", type=float, default=0.90, help="Stop if reached")
    ap.add_argument("--max-trials", type=int, default=8, help="Max trials")
    ap.add_argument("--val-split", type=float, default=0.10, help="Validation split fraction")
    ARGS = ap.parse_args()

    os.makedirs(ARGS.out, exist_ok=True)
    labels_path = os.path.join(ARGS.out, "labels.txt")
    trial_dir = os.path.join(ARGS.out, "checkpoints")
    os.makedirs(trial_dir, exist_ok=True)

    base_seed = 42

    train_ds, val_ds, class_names = make_datasets(
        ARGS.data, ARGS.img, ARGS.batch, seed=base_seed, val_split=ARGS.val_split
    )
    write_labels(labels_path, class_names)

    best = {"val_acc": 0.0, "ckpt_path": None, "trial": None, "width_mult": None, "dropout": None}

    for trial in range(1, ARGS.max_trails + 1) if hasattr(ARGS, "max_trails") else range(1, ARGS.max_trials + 1):
        model_seed = base_seed + trial * 24
        tf.keras.utils.set_random_seed(model_seed)
        random.seed(model_seed)
        np.random.seed(model_seed)

        tf.keras.backend.clear_session()
        gc.collect()

        print(f"\nTrial {trial}/{ARGS.max_trials} (seed={model_seed})")

        res = train_single_trial(
            train_ds=train_ds,
            val_ds=val_ds,
            trial_id=trial,
            trial_dir=trial_dir,
            img_size=ARGS.img,
            num_classes=len(class_names),
        )

        print(f"[trial {trial}] val_acc={res['val_acc']:.4f}  wm={res['width_mult']}  do={res['dropout']}  ckpt={os.path.basename(res['ckpt_path'])}")

        if res["val_acc"] > best["val_acc"]:
            best.update(res)
            best["trial"] = trial

        if res["val_acc"] >= ARGS.target_acc:
            print(f"\n[success] Target {ARGS.target_acc:.2f} reached in trial {trial}.")
            break

    if best["ckpt_path"] is None:
        print("\n[fail] No trials produced a checkpoint. Check dataset path.")
        sys.exit(1)

    print(f"\n[best] trial={best['trial']}  val_acc={best['val_acc']:.4f}  ckpt={best['ckpt_path']}")
    print(f"labels: {labels_path}")

if __name__ == "__main__":
    main()
