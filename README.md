# U-net
Realized by Keras+tensorflow

# 版本依赖
见requirements.txt

# 文件结构
-data
  └─ocean_sub1
    ├─results
    │  ├─prediction
    │  └─unet.hdf5
    ├─test
    └─train
      ├─image
      └─label
-main.py
-data.py
-model.py

# 预测方式
直接运行main.py即可加载训练好的模型进行识别

# 训练方式
训练模型的代码在main.py的注释中

# 致谢
https://github.com/zhixuhao/unet
