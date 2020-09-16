# 9-7 畳み込みニューラルネットワーク
## 実装
- docker-composeのruntime用にパスを通す
```
cd /etc/docker
touch daemon.json
```
- `/etc/docker/daomon.json`を編集
```
{
  "default-runtime": "nvidia",
  "runtimes": {
     "nvidia": {
       "path": "/usr/bin/nvidia-container-runtime",
       "runtimeArgs": []
       }
     }
}
```
- Docker再起動
```
sudo systemctl daemon-reload
sudo systemctl restart docker
```
- NVIDIA Dockerでgpu版tensorflowを実行
```
docker-compose build
docker-compose up -d
docker-compose exec tf_gpu /bin/bash
```
- train.pyを実行（前処理済のembeddingベクトルを読み込む）＋コンソールへ表示させながらファイルへ出力
```
/work# python -u train.py -sf -lt 2>&1 | tee hogehoge.log
```

