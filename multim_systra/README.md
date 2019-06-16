# BERT as serviceのセットアップ

https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip
からモデルをダウンロードしてmodel/に保存

```
unzip model/cased_L-24_H-1024_A-16.zip
bert-serving-start -model_dir model/cased_L-24_H-1024_A-16/ -num_worker=1
```