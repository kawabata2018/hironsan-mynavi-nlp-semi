# hironsan-mynavi-nlp-semi
> NLP社内勉強会ノート - 中山光樹『機械学習・深層学習による自然言語処理入門』

## NVIDIA Dockerの導入
### 環境
- OS: Ubuntu 20.04
```
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=20.04
DISTRIB_CODENAME=focal
DISTRIB_DESCRIPTION="Ubuntu 20.04.1 LTS"
```
- GPU: NVIDIA GeForce RTX 2080 Ti
```
26:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti Rev. A] (rev a1)
26:00.1 Audio device: NVIDIA Corporation TU102 High Definition Audio Controller (rev a1)
26:00.2 USB controller: NVIDIA Corporation TU102 USB 3.1 Host Controller (rev a1)
26:00.3 Serial bus controller [0c80]: NVIDIA Corporation TU102 USB Type-C UCSI Controller (rev a1)
```
- ドライバー: CUDA
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```

### 手順
- NVIDIA Container Toolkitをインストール
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
- NVIDIA Container Runtimeをインストール
```
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
```

> 公式
> - https://github.com/NVIDIA/nvidia-docker
> - https://github.com/NVIDIA/nvidia-container-runtime

> まとめ
> - https://qiita.com/ksasaki/items/b20a785e1a0f610efa08
> - https://techblog.cccmk.co.jp/entry/2020/03/29/094426
>

