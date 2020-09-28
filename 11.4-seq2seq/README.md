# 11-4 系列変換モデル

## モデル化
入力系列<img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathbf{\mathbb{}X}">と出力系列<img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathbf{\mathbb{}Y}">を以下のように表す。
<img src="https://latex.codecogs.com/gif.latex?\inline&space;X_i"> は系列<img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathbf{\mathbb{}X}">の<img src="https://latex.codecogs.com/gif.latex?\inline&space;i">番目の要素を表す。
- <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\large&space;\mathbf{X}&space;=&space;(x_1,&space;\dots&space;,&space;x_I)">
- <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\large&space;\mathbf{Y}&space;=&space;(y_1,&space;\dots&space;,&space;y_J)">

<img src="https://latex.codecogs.com/gif.latex?\inline&space;P(\mathbf{Y}|\mathbf{X})"> を以下のようにモデル化する。
- <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\large&space;\approx&space;P(y_1|\mathbf{X})P(y_2|y_1,&space;\mathbf{X})&space;\dots&space;P(y_{J&plus;1}|y_J,&space;\mathbf{X})">
- <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\large&space;=&space;\prod_{j=1}^{J&plus;1}&space;P(y_j|y_1,&space;\dots&space;,&space;y_{j-1}&space;,&space;\mathbf{X})">
- <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\large&space;=&space;\prod_{j=1}^{J&plus;1}&space;P(y_j|z,&space;y_1,&space;\dots&space;,&space;y_{j-1})">
- <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\large&space;=&space;\prod_{j=1}^{J&plus;1}&space;P(y_j|h_{j-1},&space;y_{j-1})">
- <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\large&space;=&space;\prod_{j=1}^{J&plus;1}&space;P(y_j|h_j)P(h_j|h_{j-1},&space;y_{j-1})">

> - <img src="https://latex.codecogs.com/gif.latex?\inline&space;z&space;=&space;Encoder(\mathbf{X})">
> - <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{120}&space;\large&space;h_j&space;=&space;Decoder(h_{j-1},&space;y_{j-1})">

## 学習
- NVIDIA Dockerでgpu版tensorflowを実行
```
docker-compose build
docker-compose up -d
docker-compose exec tf_gpu /bin/bash
```
- train.pyを実行＋コンソールへ表示させながらファイルへ出力
```
/work# python -u train.py 2>&1 | tee hogehoge.log
```

- トレーニング速度比較（Epoch平均）

|model|cpu|gpu|
|---|---|---|
|Simple seq2seq|92s|13s|

- BLEU
  - スコア：0.19±0.01
  - 算出時間：500s前後
