# 深度学习 / CITtrip项目

- [深度学习 / CITtrip项目](#%e6%b7%b1%e5%ba%a6%e5%ad%a6%e4%b9%a0--cittrip%e9%a1%b9%e7%9b%ae)
  - [CITtrip项目](#cittrip%e9%a1%b9%e7%9b%ae)
    - [资料一: 基于深度学习的网络异常检测技术研究_尹传龙](#%e8%b5%84%e6%96%99%e4%b8%80-%e5%9f%ba%e4%ba%8e%e6%b7%b1%e5%ba%a6%e5%ad%a6%e4%b9%a0%e7%9a%84%e7%bd%91%e7%bb%9c%e5%bc%82%e5%b8%b8%e6%a3%80%e6%b5%8b%e6%8a%80%e6%9c%af%e7%a0%94%e7%a9%b6%e5%b0%b9%e4%bc%a0%e9%be%99)
    - [资料三: A Survey on Deep Learning: Algorithms, Techniques, and Applications 深度学习概览：算法，技术和应用](#%e8%b5%84%e6%96%99%e4%b8%89-a-survey-on-deep-learning-algorithms-techniques-and-applications-%e6%b7%b1%e5%ba%a6%e5%ad%a6%e4%b9%a0%e6%a6%82%e8%a7%88%e7%ae%97%e6%b3%95%e6%8a%80%e6%9c%af%e5%92%8c%e5%ba%94%e7%94%a8)
    - [应用一：Tensorflow 使用](#%e5%ba%94%e7%94%a8%e4%b8%80tensorflow-%e4%bd%bf%e7%94%a8)

> 参考资料 ：CITtrip
>
> 2.1 更新：《深度学习概览：算法，技术和应用》论文学习

## CITtrip项目

> 题目：基于深度学习的网络异常行为检测

### 资料一: 基于深度学习的网络异常检测技术研究_尹传龙

1. 绪论

   - 异常行为检测概述
     - 网络异常：网络设备异常、网络安全异常
     - 异常检测：设定正常行为轮廓“阈值”
       - 入侵检测：违背既定安全策略
       - 僵尸网络检测：受感染计算机构成的网络
   - 研究现状
     - 统计分析、数据挖掘、特征工程、
     - 机器学习
       - 贝叶斯网络
       - 遗传算法：模拟选择、交叉、变异等操作的搜索方法
       - 支持向量机：寻找最佳超平面区分不同类的样本
       - k最近邻
       - 决策树
       - 模糊逻辑
       - 人工神经网络
       - 深度学习
   - 数据集
     - 检测方法在实际中的检测性能效果高度依赖数据集
  
2. 全连接循环神经网络的入侵检测技术(略)

3. 生成式对抗网络的入侵检测技术

   - 生成式对抗网络：目的是学习真实数据样本的概率分布
     - 深度生成模型、引入对抗思想，解决样本较少问题
     - 由生成模型和判别模型组成
       - 生成模型为了尽可能模仿真实样本，学习捕捉真实数据样本的概率分布
       - 判别模型对真实样本和生成样本分类，等价于二分类模型
       - 双方对抗、优化，达到纳什平衡：生成
     - 问题：基于极大极小博弈机制的优化问题
       - 一方面：进化判别模型D，提升判别准确率
         - D(x) 将真实样本x判断为真实样本的概率
         - D(G(z)) 将生成样本判断为真实样本的概率
         - 目的：改善D，使一增大二减小
       - 另一方面：进化生成模型G，尽量生成与真实数据相似的样本
         - G(z) 噪声z到生成样本空间的映射

### 资料三: A Survey on Deep Learning: Algorithms, Techniques, and Applications 深度学习概览：算法，技术和应用

> Deep learning uses multiple layers to represent the abstractions of data to build computational models.
>
> The lack of core understanding renders these powerful methods as black-box machines that inhibit development at a fundamental level.
>
> Moreover, deep learning has repeatedly been perceived as a silver bullet to all stumbling blocks in machine learning, which is far from the truth.

- INTRODUCTION
  - 应用：多媒体概念检索，图像分类，视频推荐，社交网络分析，文本挖掘，自然语言处理（NLP），视觉数据处理，语音和音频处理
  
    Deep learning, which has its roots from conventional neural networks, significantly outperforms its predecessors. It utilizes graph technologies with transformations among neurons to develop many-layered learning models.
    深度学习起源于传统的神经网络，其性能明显优于其前辈。它利用图技术以及神经元之间的转换来开发多层学习模型。
  - feature engineering 传统上，不良的数据表示通常会导致性能降低，因此，特征工程一直是重要的，且局限在特定领域，需要大量的人力
  - 相比之下，深度学习算法以自动方式执行特征提取，

    These algorithms include a layered architecture of data representation, where the high-level features can be extracted from the last layers of the networks while the low-level features are extracted from the lower layers这些算法包括分层的数据表示架构，高层特征可以从网络的最后一层中提取，而底层要素从较低层提取

    The input is the scene information received from eyes, while the output is the classified objects.我们的大脑可以自动从不同场景中提取数据表示。 输入是从眼睛接收场景信息，而输出是分类对象。

  - History
    - In 1980, neocogitron was intrduced, which inspired the convolutional neural network
    - Recurrent Neutal Networks were proposed in 1986
    - LeNet made the Deep Neural Networks
    - Deep Belief Networks (DBNs):Its main idea was to train a simple two-layer unsupervised model like Restricted Boltzmann Machines (RBMs), freeze all the parameters, stick a new layer on top, and train just the parameters for the new layer. 它的主要思想是训练一个简单的两层无监督模型，例如Restricted玻尔兹曼机器（RBM），冻结所有参数，在顶部粘贴新层并仅训练新层的参数。
    - deep learning now is one of the most efficient tools compared to other machinelearning algorithms with great performanc.与其他机器学习相比，深度学习现在是最有效的工具之一
    - ***从最初的Artificial Neural Networks(ANN)，到Deep Belief Networks DBN，Restricted Boltzmann Machines (RBMs)，Recurrent Neural Networks (RNNs)和Convolutional Neural Networks（CNN）***
    - 由于大量数据没有标签或带有噪音标签，因此一些研究更多地侧重于使用**无监督或半监督的**深度学习技术来提高训练模块的噪声鲁棒性。
    - cross-modality structure：由于当前大多数深度学习模型仅关注单一模式，因此导致对真实数据的表示有限。 研究人员现在更加关注跨模式结构
    - **Google AlphaGo**
  - Research Objectives and Outline 研究目的和概述
    - 介绍顶级论文，作者的经验，以及在深度神经网络研究和应用中的突破。
    - 在我们的调查中，提出了深度学习关键领域中的挑战和机遇，包括并行性，可伸缩性，功能和优化。为了解决上述问题，在不同的领域中引入了不同种类的深度网络，例如用于NLP的RNN和用于图像处理的CNN。 本文还介绍并比较了流行的深度学习工具，包括**Caffe，DeepLearning4j，TensorFlow，Theano和Torch**，以及每种深度学习工具中的优化技术。

- DEEP LEARNING NETWORKS：在第2节中，简要介绍了流行的深度学习网络。
  - 本节介绍的深度学习网络与其要点
      | DL Networks | Descriptive Key Points                            |
      | ----------- | ------------------------------------------------- |
      | RvNN        | 使用树状结构。 NLP的首选                          |
      | RNN         | 适合顺序信息。首选用于NLP和语音处理               |
      | CNN         | 最初用于图像识别，扩展到NLP，语音处理和计算机视觉 |
      | DBN         | 无监督学习。定向连接                              |
      | DBM         | 无监督学习。RBM的复合模型。无向连接               |
      | GAN         | 无监督学习。博弈论框架                            |
      | VAE         | 无监督学习。概率图形模型                          |
  - Recursive Neural Network (RvNN)
  - Recurrent Neural Network (RNN)
  - Convolutional Neural Network (CNN)
  - Deep Generative Networks

- DEEP LEARNING TECHNIQUES AND FRAMEWORKS：第三部分讨论了深度学习中的几种算法，技术和框架。
  - Unsupervised and Transfer Learning 无监督和转移学习
  - Online Learning
  - Optimization Techniques in Deep Learning 深度学习中的优化技术
  - Deep Learning in Distributed Systems
  - Deep Lear ning Frameworks
- VARIOUS APPLICATIONS OF DEEP LEARNING：第4节中提供了许多深度学习的应用。
  - Natural Language Processing（略）
  - Visual Data Processing（略）
  - Speech and Audio Processing（略）
  - Other Applications（略）
- DEEP LEARNING CHALLENGES AND FUTURE DIRECTIONS：第5节指出了未来的挑战和潜在的研究方向。

### 应用一：Tensorflow 使用

[Essential_documentation](https://tensorflow.google.cn/guide/)

1. 安装
   1. 网址 [TensorFlow](https://tensorflow.google.cn/)
   2. 软件需求：python, pip, virtualenv\conda
   3. 首次安装的版本可能并不契合网络资源所要求的tensorflow及其所需的python版本，因此可以使用conda创建虚拟环境选择合适的版本。
   4. 步骤
      - 下载anaconda_package（包含图形界面，对新手较友好）

      ``` shell
      > conda --version
      ```

      - 打开terminal，挂上SJTU_vpn
      - 通过conda创建虚拟环境并初始化python，激活刚刚创建的虚拟环境

      ``` shell
      > conda create -n venv pip python=3.7
      > conda activate venv
      ```

      ``` shell
      > conda create -n tensorflow3.5 python=3.5
      > conda activate tensorflow3.5
      ```

      - 通过pip包管理器，从网址[package location](https://tensorflow.google.cn/install/pip#package-location)安装所需的tensorflow包

      ``` shell
      (venv) > pip install --ignore-installed --upgrade (packageURL)
      (venv) > pip install --upgrade tensorflow
      ```

2. 使用
   1. 步骤
      - 激活环境（也可以通过anaconda图形界面打开终端），进入pyton

        ``` shell
        > conda activate venv # 激活虚拟环境
        > python
        > conda deactivate #退出虚拟环境
        ```

      - import tensorflow，导入tf.keras

          ```python
          import tensorflow as tf
          from tensorflow.keras import layers
          print(tf.__version__)
          print(tf.keras.__version__)
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

      如果您没有GPU并且希望尽可能多地利用CPU，那么如果您的CPU支持AVX，AVX2和FMA，则应该从针对CPU优化的源构建tensorflow。

      暂时解决方案：初学阶段重点在于理解原理，对于效率的提升不是重点，故忽略它（嘿嘿）

      每次进入python环境时执行下列代码段，降低报警频率。

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
