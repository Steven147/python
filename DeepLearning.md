# 深度学习 / CITtrip项目 /

> 参考资料

大学课业/CITtrip

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
  
2. 全连接循环神经网络的入侵检测技术

3. 生成式对抗网络的入侵检测技术

   * 生成式对抗网络：学习真实数据样本的概率分布
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

1. 安装
   1. 网址 [TensorFlow](https://tensorflow.google.cn/)
   2. 软件需求：python, pip, virtualenv\conda
   3. 首次安装的版本可能并不契合网络资源所要求的tensorflow和python版本，因此可以使用conda创建虚拟环境选择合适的版本。

   ``` shell
   > conda create -n venv pip python=3.7  # select python version
   > source activate venv
   (venv) > pip install --ignore-installed --upgrade (packageURL)
   (venv) > pip install --upgrade tensorflow
   (venv) > source deactivate
   ```

      [package location](https://tensorflow.google.cn/install/pip#package-location)
2. 使用
   1. 
