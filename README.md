# たのしくないdeepfaceの環境構築
## はじめに
deepfaceは様々な研究で得られたstate-of-the-artな顔照合学習モデルの評価を簡便に行えるラッパー的な軽量フレームワークです。VGG-Face, OpenFace, Google FaceNet, Facebook DeepFace, Dlib, DeepID, ArcFace, SFace, GhostFaceNetを一度に評価できるのがウリです。

ただ、ですね…、こういうリポジトリにありがちなのですが、年数が経っているのでそのままでは動作しないわけです。

いや、ちょっとした変更で動作はするんですけどGPUを利用できないとかふつうに言ってきます。それどころか親切に用意してあるPyPIからのインストールやDockerfileを利用すると動作しない、setup.pyに記載されているPythonバージョンがDockerfileに記載されているものと違う、というトラップも抜かりありません。

それでもまぁ「行けるだろ」と鼻をほじりながらいろいろ試して半日の時間を吸われました。[^1]

[^1]: 本来ならリポジトリオーナーへのリスペクトと同時にISSUEなど投げるのがあるべき姿なのですが、「ちょっとなんでこうしてるのか分かんない…」ところが多々あったので、そっとしておくべきと判断致しました…。

もったいないので変更点をシェアして供養します。

## 最初に結論
色々やったのですが、最終的には「自分でDockerfile書いてコンテナの中で処理させる」が正解です。Python仮想環境内で、とか、deepFaceが用意してくれているDockerfileをそのまま使うとかだと失敗します。内部でツギハギのコードがあって、その出所が古い箇所が依存関係で荒ぶりまくります。
一つ注意点なのですが、可能性は低いですがもしかするとホスト環境によってはこのDockerfileのままだとCUDAまわりが荒ぶるかも知れません。[^2]

[^2]: そのときはcuda-toolkitやlibcublasのバージョンを修正してください。ただし下手にいじると内部のツギハギコードが荒ぶります…。

まず、プロジェクトルートディレクトリに最新のコードをダウンロードしておいてください。
```bash
git clone https://github.com/serengil/deepface.git
```

このままだとプロジェクトルート/deepface/deepface/...となりますので、プロジェクトルート/deepface/...にしておいてください。

`venv/lib/python3.10/site-packages/deepface/commons/image_utils.py`において、97行目から99行目にかけて、日本語パス対応でエラーを発生させる箇所がありますが、ここをコメントアウトします。[^3]

[^3]: パスにascii以外が含まれていると即停止させる謎仕様です…。

```diff
--- a/venv/lib/python3.10/site-packages/deepface/commons/image_utils.py
+++ b/venv/lib/python3.10/site-packages/deepface/commons/image_utils.py
@@ -97,7 +97,7 @@
    # image name must have english characters
-    if not img.isascii():
-        raise ValueError(f"Input image must not have non-english characters - {img}")
+    # if not img.isascii():
+    #     raise ValueError(f"Input image must not have non-english characters - {img}")
```
修正したら以下をDockerfileとして保存します。

```bash: Dockerfile
# base image
FROM python:3.8.12
LABEL org.opencontainers.image.source https://github.com/serengil/deepface

# -----------------------------------
# create required folder
RUN mkdir -p /app && chown -R 1001:0 /app
RUN mkdir /app/deepface

# -----------------------------------
# switch to application directory
WORKDIR /app

# -----------------------------------
# update image OS and install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------
# Install CUDA and cuDNN
RUN distribution=$( . /etc/os-release; echo $ID$VERSION_ID) && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-11-5 libcudnn8 libcudnn8-dev libcublas-12-0 && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV PATH=/usr/local/cuda-11.5/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# -----------------------------------
# Copy required files from repo into image
COPY ./deepface /app/deepface
COPY ./requirements.txt /app/requirements.txt
COPY ./requirements_local /app/requirements_local.txt
COPY ./package_info.json /app/
COPY ./setup.py /app/
COPY ./README.md /app/
COPY ./entrypoint.sh /app/deepface/api/src/entrypoint.sh

# -----------------------------------
# install dependencies - deepface with these dependency versions is working
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -r /app/requirements_local.txt
# install deepface from source code (always up-to-date)
RUN pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org -e .

# Install TensorFlow compatible with CUDA setup
RUN pip uninstall -y tensorflow keras && \
    pip install --no-cache-dir tensorflow==2.13.1  && \
    pip install matplotlib && \
    pip install scikit-learn
# -----------------------------------
# environment variables
ENV PYTHONUNBUFFERED=1

# -----------------------------------
# set working directory for runtime
WORKDIR /app/deepface
```

おもむろにビルド。
```bash: docker build
docker build --network host --no-cache -t deepface-gpu .
```

コンテナ起動。
```bash
docker run --gpus all --network host -it \
-v <プロジェクトルートパス>:/app/deepface \
deepface-gpu /bin/bash
```

### テスト1
```bash
root@terms:/app/deepface# python
Python 3.8.12 (default, Mar  2 2022, 04:56:27) 
[GCC 10.2.1 20210110] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2025-01-01 05:29:24.125778: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-01-01 05:29:25.281817: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
>>> print(tf.config.list_physical_devices('GPU'))
2025-01-01 05:29:32.840290: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2025-01-01 05:29:32.916196: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
2025-01-01 05:29:32.920533: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:995] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>> 
```
このように出力されれば成功。

### テスト2
プロジェクトルートにdeepface_test.pyを用意します。
```python: deepface_test.py
from deepface import DeepFace

img1 = "assets/あいう/a.png"
img2 = "assets/あいう/b.png"

result = DeepFace.verify(
    img1_path=img1,
    img2_path=img2,
    model_name="VGG-Face",
    distance_metric="cosine"
)
print("Is verified: ", result["verified"])
```
GPUを使用しつつ結果が出力されれば成功です。



