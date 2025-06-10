import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from utils.mnist_reader import load_mnist  # 用於 model_test.py 的推論

# === 使用 TensorFlow 官方訓練集來訓練 ===
(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()

# === 使用原始 `t10k` 資料作為測試集（保證 model_test 可測） ===
x_test, y_test = load_mnist('./data/fashion', kind='t10k')

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

# === 建立模型 ===
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# 編譯與訓練
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

model.fit(x_train, y_train,
          epochs=10,
          validation_split=0.1,
          callbacks=[early_stop])

# === 使用原始 t10k 測試準確度 ===
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# === 儲存模型為 .h5（非必要給 model_test 用）===
model.save("fashion_mnist.h5")

# === 儲存 JSON 結構 & 權重給 model_test 用 ===
from collections import OrderedDict
import pickle

# 建構 dict 權重
weights = OrderedDict()
model_config = model.get_config()
for layer in model_config["layers"]:
    lname = layer["config"]["name"]
    if "Dense" in layer["class_name"]:
        w, b = model.get_layer(name=lname).get_weights()
        weights[f"{lname}/kernel:0"] = w
        weights[f"{lname}/bias:0"] = b
    elif "Flatten" in layer["class_name"]:
        weights[f"{lname}/kernel:0"] = np.array([])  # 為避免 model_test crash

    layer["weights"] = (
        [f"{lname}/kernel:0", f"{lname}/bias:0"]
        if "Dense" in layer["class_name"]
        else []
    )

# 儲存成 npz（但先存成 dict，避免 lazy NpzFile 行為）
np.savez("model/fashion_mnist.npz", **weights)

# 修改 JSON 結構（加上 weights 欄位）以供 model_test.py 使用
import json
model_config = model.get_config()
for layer in model_config["layers"]:
    lname = layer["config"]["name"]
    if "Dense" in layer["class_name"]:
        layer["weights"] = [f"{lname}/kernel:0", f"{lname}/bias:0"]
    elif "Flatten" in layer["class_name"]:
        layer["weights"] = []

model_arch = {
    "class_name": "Sequential",
    "config": model_config["layers"]
}
with open("fashion_mnist.json", "w") as f:
    json.dump(model_arch, f)
