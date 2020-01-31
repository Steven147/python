# 深度学习 / CITtrip项目 

> 参考资料 ：CITtrip

## CITtrip项目

> 题目：基于深度学习的网络异常行为检测

### 1 基于深度学习的网络异常检测技术研究_尹传龙

1. 绪论

   * 异常行为检测概述
     * 网络异常：网络设备异常、网络安全异常
     * 异常检测：设定正常行为轮廓“阈值”
       * 入侵检测：违背既定安全策略
       * 僵尸网络检测：受感染计算机构成的网络
   * 研究现状
     * 统计分析、数据挖掘、特征工程、
     * 机器学习
       * 贝叶斯网络
       * 遗传算法：模拟选择、交叉、变异等操作的搜索方法
       * 支持向量机：寻找最佳超平面区分不同类的样本
       * k最近邻
       * 决策树
       * 模糊逻辑
       * 人工神经网络
       * 深度学习
   * 数据集
     * 检测方法在实际中的检测性能效果高度依赖数据集
  
2. 全连接循环神经网络的入侵检测技术(略)

3. 生成式对抗网络的入侵检测技术

   * 生成式对抗网络：目的是学习真实数据样本的概率分布
     * 深度生成模型、引入对抗思想，解决样本较少问题
     * 由生成模型和判别模型组成
       * 生成模型为了尽可能模仿真实样本，学习捕捉真实数据样本的概率分布
       * 判别模型对真实样本和生成样本分类，等价于二分类模型
       * 双方对抗、优化，达到纳什平衡：生成
     * 问题：基于极大极小博弈机制的优化问题
       * 一方面：进化判别模型D，提升判别准确率
         * D(x) 将真实样本x判断为真实样本的概率
         * D(G(z)) 将生成样本判断为真实样本的概率
         * 目的：改善D，使一增大二减小
       * 另一方面：进化生成模型G，尽量生成与真实数据相似的样本
         * G(z) 噪声z到生成样本空间的映射

### Tensorflow 使用

[Essential_documentation](https://tensorflow.google.cn/guide/)

1. 安装
   1. 网址 [TensorFlow](https://tensorflow.google.cn/)
   2. 软件需求：python, pip, virtualenv\conda
   3. 首次安装的版本可能并不契合网络资源所要求的tensorflow及其所需的python版本，因此可以使用conda创建虚拟环境选择合适的版本。
   4. 步骤
      * 下载anaconda_package（包含图形界面，对新手较友好）

      ``` shell
      > conda --version
      ```

      * 打开terminal，挂上SJTU_vpn
      * 通过conda创建虚拟环境并初始化python，激活刚刚创建的虚拟环境

      ``` shell
      > conda create -n venv pip python=3.7
      > conda activate venv
      ```

      ``` shell
      > conda create -n tensorflow3.5 python=3.5
      > conda activate tensorflow3.5
      ```

      * 通过pip包管理器，从网址[package location](https://tensorflow.google.cn/install/pip#package-location)安装所需的tensorflow包

      ``` shell
      (venv) > pip install --ignore-installed --upgrade (packageURL)
      (venv) > pip install --upgrade tensorflow
      ```

2. 使用
   1. 
   
   
## A Survey on Deep Learning: Algorithms, Techniques, and Applications 深度学习概览：算法，技术和应用
   1. 步骤
      * 激活环境（也可以通过anaconda图形界面打开终端），进入pyton

      ``` shell
      > conda activate venv # 激活虚拟环境
      > python
      > conda deactivate #退出虚拟环境
      ```

      * import tensorflow，导入tf.keras

          ```python
          import tensorflow as tf
          from tensorflow.keras import layers
          print(tf.__version__)
          print(tf.keras.__version__)
          ```

      ```python
      import tensorflow as tf 
      tf.compat.v1.disable_eager_execution() 
      sess=tf.compat.v1.Session() 
      hello= tf.constant('Hello, TensorFlow!') 
      print(sess.run(hello))
      ```

3. 问题修复

   1. >"Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"

      ```python
        >>> w=tf.Variable([[0.5,1.0]])
        2020-01-31 00:15:50.511182: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
        2020-01-31 00:15:50.527003: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f7fc1dcc1a0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
        2020-01-31 00:15:50.527021: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
      ```

      问题所在：

      [彻底解决“Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA”警告](https://blog.csdn.net/wlwlomo/article/details/82806118)

      [警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA](https://blog.csdn.net/hq86937375/article/details/79696023)

      链接中提到，要换成支持cpu用AVX2编译的TensorFlow版本。

      > 如果您没有GPU并且希望尽可能多地利用CPU，那么如果您的CPU支持AVX，AVX2和FMA，则应该从针对CPU优化的源构建tensorflow。

      暂时解决方案：忽略它（嘿嘿）

      每次进入python环境时执行下列代码段

      ```python
      import os
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
      ```

   2. 版本问题

      ```python
      raise RuntimeError('The Session graph is empty.  Add operations to the '
      RuntimeError: The Session graph is empty.  Add operations to the graph before calling run().
      ```

      暂时结论： tensorflow版本问题，函数使用方式产生了变更，一种方法是下载对应版本的tf，一种方法是改进使用代码。
