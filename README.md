# deepFace:独自DockerコンテナでGPUから利用する
## はじめに
deepFaceは様々な研究で得られたstate-of-the-artな顔照合学習モデルの評価を簡便に行えるラッパー的な軽量フレームワークです。VGG-Face, OpenFace, Google FaceNet, Facebook DeepFace, Dlib, DeepID, ArcFace, SFace, GhostFaceNetを一度に評価できるのがウリです。（顔検出はまた別）

ただ、ホスト環境によって動作しないことってありますよね。
動作はするけどGPUを利用できない、とか。[^0]

[^0]: 親切に用意してあるPyPIからのインストールやDockerfileを利用すると動作しない、setup.pyに記載されているPythonバージョンがDockerfileに記載されているものと違う、というトラップも。残念ながらdeepFaceでこの状況にハマってしまいました。

それでもまぁ「なんとかなるべ」と鼻をほじりながらいろいろ試して半日の時間を吸われました。[^1]

[^1]: おそらくですがUbuntu 20.04、のようなちょっと古めのホストなら何の問題も無いと思います。また本来ならリポジトリオーナーへのリスペクトと同時にISSUEなど投げるのがあるべき姿なのですが「なんでこうしてるのか分かんない」ところがあって、門外漢が口を出すべきでないと判断致しました。似てる内容のISSUEも上がってましたし。

`pip install deepface`でGPU利用できない方々に向けて、変更点をシェアして供養します。

![](assets/eye-catch.png)

## 環境
```bash
# pyenvによりpythonバージョン変更済み
user@user:~/<プロジェクトルートディレクトリ>$ python -V
Python 3.8.12
user@user:~/<プロジェクトルートディレクトリ>$ inxi -SG --filter
System:
  Kernel: 6.8.0-50-generic x86_64 bits: 64 Desktop: GNOME 42.9
    Distro: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
Graphics:
  Device-1: NVIDIA TU116 [GeForce GTX 1660 Ti] driver: nvidia v: 555.42.06
  Display: x11 server: X.Org v: 1.21.1.4 driver: X: loaded: nouveau
    unloaded: fbdev,modesetting,vesa failed: nvidia gpu: nvidia
    resolution: 2560x1440~60Hz
  OpenGL: renderer: NVIDIA GeForce GTX 1660 Ti/PCIe/SSE2
    v: 4.6.0 NVIDIA 555.42.06
```

## 最初に結論
おそらくですがUbuntu 20.04、のようなちょっと古めのホストなら何の問題も無いと思います。それ以外のホスト環境で、ということで以下お願いします。

色々やったのですが、最終的には「自分でDockerfile書いてコンテナの中で処理させる」が正解でした。Python仮想環境内かつpyenvで、とか、deepFaceが用意してくれているDockerfileをそのまま使うとかやったのですが失敗しました。

可能性は低いですがホスト環境によってはこのDockerfileのままだとCUDAまわりが荒ぶるかも知れません。そのときはcuda-toolkitやlibcublasのバージョンを修正してください。[^2]

[^2]: ただし下手にいじるとtensorflowまわりが荒ぶります。

まず、プロジェクトルートディレクトリに最新のコードをダウンロードしておいてください。
```bash
git clone https://github.com/serengil/deepface.git
```

このままだとプロジェクトルート/deepface/deepface/...となりますので、プロジェクトルート/deepface/...にしておいてください。

`deepface/commons/image_utils.py`の97行目から99行目にかけて、日本語パスでエラーを発生させる箇所がありますが、ここをコメントアウトします。[^3]

[^3]: パスにascii以外が含まれていると即停止させる謎仕様です。

```diff
--- a/deepface/commons/image_utils.py
+++ b/deepface/commons/image_utils.py
@@ -97,7 +97,7 @@
    # image name must have english characters
-    if not img.isascii():
-        raise ValueError(f"Input image must not have non-english characters - {img}")
+    # if not img.isascii():
+    #     raise ValueError(f"Input image must not have non-english characters - {img}")
```
修正したら以下をDockerfileとして保存します。[^4]

[^4]: オリジナルのDockerfileを改変したものです。オリジナルはサーバを起ち上げ時にエラーが発生します。これらを改変してGPU対応にしたものです。

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
root@user:/app/deepface# python
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
GPUを使用しつつ結果が出力されれば成功です。必要な重みファイルは自動的にダウンロードされます。

## おわりに
deepFaceそれ自体は2024年も手が加えられていますし、実際に使用して記事をアップしておられる方もいらっしゃるのですね。ただpytorchと比べ、tensorflowを内部で使用してるリポジトリは荒ぶりやすいように感じます（pytorchが内部でよろしくやってくれるのが大きい）。
決してdeepFaceの使い勝手がよろしくないわけではないことを記載しておきます。

さてGPUも使えるようになったので別記事用の実験をしています。かなり重い処理のはずですがGPUメモリは5.6GB程度で済んでいます。

`pip install deepface`でGPU利用できなかったらこの記事を参考にしてください。

以上です。ありがとうございました。


