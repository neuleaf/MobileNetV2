# MobileNet V2 with Tensorflow
论文链接 https://arxiv.org/abs/1801.04381
## Task1 classification
1. 文件路径及标签
   训练图片及类别标签组织为`train_data.txt`的形式，每一行表示一个样本及其标签，用空格或Tab隔开.
2. 运行可以参考`run.sh`，写入必要的参数，如类别个数、训练图片目录等.
3. test/infer

    测试时需要指定`--no_train`，如：
   `python main.py --no_train`

## Task2 detection

TODO
## Task3 segmentation

TODO