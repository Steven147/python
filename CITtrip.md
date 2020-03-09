# CITtrip项目：基于深度学习的网络异常行为检测

> 2.1 更新：《深度学习概览：算法，技术和应用》论文学习
>
> 2.2 更新：吴恩达-深度学习-bilibili
>
> 2.18 更新：《网络异常检测技术概述》论文学习 + tensorflow_keras学习
>
> 2.19 更新：《生成式对抗网络GAN的研究进展与展望_王坤峰》论文学习 + GAN模型损失函数理解 + tensorflow_keras学习2
>
> 2.20 更新：《基于深度卷积神经网络的网络流量分类方法》《卷积神经网络研究综述_周飞燕》论文学习 + tensorflow_keras学习3
>
> 2.21 更新：基于 tensorflow keras 的卷积神经网络实现 + tensorflow_gpu_for_mac install
>
> 2.22 更新：《基于深度置信网络的入侵检测模型》论文学习 + tensorflow keras 学习
>
> 2.23 更新：tensorflow keras
>
> 2.25 更新：《基于深度学习的网络流量分类及异常检测方法研究_王伟》 + 网络流量dataset组成 学习
>
> 2.26 更新：反向传播算法bp理解
>
> 2.29 更新：tensorflow keras 学习
>
> 3.6 更新：本地程序运行 用tf.keras实现GAN网络

## 信息安全竞赛安排

## 资料一: 基于深度学习的网络异常检测技术研究_尹传龙

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
         - D(x) 将真实样本x判断为真实样本的概率，正确判断
         - D(G(z)) 将生成样本判断为真实样本的概率，错误判断
         - 目的：改善D，使D(x)增大、D(G(z))减小
       - 另一方面：进化生成模型G，尽量生成与真实数据相似的样本
         - G(z) 噪声z到生成样本空间的映射
         - 目的：改善G，使D(G(z))增大

## 资料二: A Survey on Deep Learning: Algorithms, Techniques, and Applications 深度学习概览：算法，技术和应用

> Deep learning uses multiple layers to represent the abstractions of data to build computational models.
>
> The lack of core understanding renders these powerful methods as black-box machines that inhibit development at a fundamental level.
>
> Moreover, deep learning has repeatedly been perceived as a silver bullet to all stumbling blocks in machine learning, which is far from the truth.

- INTRODUCTION
  - 应用：多媒体概念检索，图像分类，视频推荐，社交网络分析，文本挖掘，**自然语言处理（NLP）**，视觉数据处理，语音和音频处理
  
    Deep learning, which has its roots from conventional neural networks, significantly outperforms its predecessors. It utilizes graph technologies with transformations among neurons to develop many-layered learning models.
    深度学习起源于传统的神经网络，其性能明显优于其前辈。它利用图技术以及神经元之间的转换来开发多层学习模型。
  - feature engineering 传统上，不良的数据表示通常会导致性能降低，因此，特征工程一直是重要的，且局限在特定领域，需要大量的人力
  - 相比之下，深度学习算法以自动方式执行特征提取，

    These algorithms include a layered architecture of data representation, where the high-level features can be extracted from the last layers of the networks while the  low-level features are extracted from the lower layers这些算法包括分层的数据表示架构，高层特征可以从网络的最后一层中提取，而底层要素从较低层提取

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

## 资料三：A survey of network anomaly detection techniques 网络异常检测技术概述

1. 介绍
   1. 异常检测：数据挖掘任务 又称 utlier detection, novelty detection，deviation detection and exception mining.
   2. 异常：一个`与众不同`、疑似是`由不同的机制所产生`的观察结果
   3. 研究的挑战：异常检测技术普适性差（有线网-无线网）、噪声、缺乏labeled dataset
   4. 标准：计算复杂度、攻击检测优先级、输出
2. 前言
   1. 异常类型
      1. 点异常：
      2. 上下文异常：圣诞节交通量高在上下文中是正常的
      3. 集体异常：低心电图
   2. 异常检测技术的输出
      1. score
      2. label
   3. 网络攻击的类型
      1. 拒绝服务DoS攻击：server is flooded with numerous connection requests.
      2. 探针：探查攻击，对主机和网络攻击的第一步
      3. 用户到根U2R：获取super账户访问权限
      4. 远程用户R2U：
   4. 对异常的网络攻击进行分类  
      1. DoS：集体异常  
      2. 探测攻击：上下文异常
      3. 用户到根U2R，远程用户R2U：点异常
3. 基于分类的网络异常检测
   1. 支持向量机
      1. 寻找最优超平面，最大化分离
      2. `监督学习`！，
      3. 实例：正常活动下，有一组注册表项会被访问
   2. 贝叶斯网络
   3. 神经网络
   4. 基于规则：学习系统的正常行为
4. 统计异常检测
   1. 混合模型：数据由两个分布M，A其中一个生成，由A生成则是异常数据
   2. 信号处理技术
   3. 主成分分析：无假设、降维且不损失重要信息
5. 信息理论
   1. 信息熵、条件熵、相对熵
   2. 相关分析：网络流量间的非线性相关性
6. 聚类：无监督！
   1. 常规聚类：k-means聚类算法，分为正常，异常两类
7. 入侵检测数据集：
   1. DARPA / KDD数据集
   2. 当代网络攻击评估数据集：ADFA-LD12
8. 评估

## 资料四：生成式对抗网络GAN的研究进展与展望_王坤峰

1. GAN提出背景
   1. 生成式模型：通过生成方法（对数据分布的学习）得到的模型
      1. `人类理解数据角度`：对数据进行分布/参数假设-用真实数据进行训练/拟合-用学习到的分布/模型生成新样本：
         1. 最大似然估计
         2. **马尔科夫链**：效率较低
      2. `机器理解数据角度`：模型不直接拟合，而是从未明确假设的分布（初始数据集）中获取数据，对模型进行修正
   2. 神经网络：解决参数多、训练难的问题；
      1. 训练：反向传播算法
      2. 结构：自由灵活
      3. 建模能力：理论上能够逼近任意函数
   3. 对抗思想：AlphaGo 两个网络博弈
2. GAN的理论与实现模型
   1. 基本原理
      1. 生成器：学习真实的数据分布
         - G(z) 噪声z到生成样本空间的映射
         - 目的：改善G，使D(G(z))增大
      2. 判别器：判别数据是否源于生成器
         1. 判断是真实数据, 标记为 1（和D的取值不同！！
         2. 判断是生成器数据，标记为 0（和D的取值不同！！
         - D(x) 将真实样本x判断为真实样本的概率，正确判断
         - D(G(z)) 将生成样本判断为真实样本的概率，错误判断
         - 目的：改善D，使D(x)增大、D(G(z))减小
   2. `学习方法！！`：
      1. 最优化判别器：损失函数：对照图像理解

          > logistic回归 损失函数
          >
          > $J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{{Cost}\left( {h_\theta}\left( {x}^{\left( i \right)} \right),{y}^{\left( i \right)} \right)}$
          >
          > 为得到凸函数$J\left( \theta  \right)$
          >
          > $J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$

         - 训练判别器的损失函数：自变量有两个来源
           - ${D_\theta}\left( {x}\right)$ 正确判断概率，目标为1，损失即与1的距离：$\log \left( {D_\theta}\left( {x} \right) \right)$
           - ${D_\theta}\left( g\left(z \right)\right)$ 错误判断概率，目标为0，损失即与0的距离：$\log \left(1- {D_\theta}\left( {z} \right) \right)$
         - $J\left( \theta_D \right)=-\frac{1}{2}{E_1}\log \left( {D_\theta}\left( {x} \right) \right)- \frac{1}{2}E_2 \log \left( 1-{D_\theta }\left( G\left(z \right)\right) \right)$
  
      2. 最优化生成器：损失函数，取判别器损失函数的相反数
         - $J\left( \theta_G \right)=\frac{1}{2}{E_1}\log \left( {D}\left( {x} \right) \right)+ \frac{1}{2}E_2 \log \left( 1-{D}\left[ G_\theta\left(z \right)\right] \right)$

      3. 过程：采用交替优化的方法：
         - $max_D min_G\{\ Loss\left(\theta_D \theta_G \right)={E_1}\log \left( {D_\theta}\left( {x} \right) \right)+ E_2 \log \left( 1-D_\theta \left[ G_\theta\left(z \right)\right] \right)\ \}$
         - 先固定生成器 G, 优化判别器 D, 使得 D 的判别准确率最大化; 
         - 然后固定判别器 D, 优化生成器 G, 使得 D 的判别准确率最小化
      4. GAN衍生模型
3. GAN的应用领域
   1. 视觉
   2. 语音语言
   3. 其他·
4. GAN的思考和展望
   1. 作为一种`生成式方法`, 有效解决了可建立自然性解释的数据的生成难题
   2. 将两个神经网络的对抗作为训练准则并且可以使用反向传播进行训练, 训练过程不需要效率较低的马尔科夫链方法, 也不需要做各种近似推理, 没有复杂的变分下界,

## 资料五：基于深度卷积神经网络的网络流量分类方法

1. 引言
   1. 在网络流量分类领域的机器学习算法可以分为浅层学习和深度学习
   2. 浅层学习主要包括:支持向量机、决策树、 贝叶斯和 k-means
   3. 深度学习主要包括`**深度置信网络**`、`**卷积神经网络**`和`**递归神经网络**`等
   4. 基于离差标准化的卷积神经网络 min-max normalization convolutional neural network
2. 流量分类问题及描述
   1. 总体流程包括
      1. 网络数据的采集 现有的数据集 网络协议数据分析工具
      2. 带有准确背景信息数据集的生成
      3. 数据集的预处理，
      4. 流量特征的提取以及分类
   2. 采用卷积神经网络，构造合适的特征空间
      1. 对数据集进行数据预处理：将网络流量数据转化为灰度图片
      2. 将灰度图片作为卷积神经网络的输入数据进行学习
3. 基于改进卷积神经网络的网络流量分类方法
   1. 网络流量数据集的构建
      1. Moore 数据集
      2. 实际数据集
   2. 数据预处理
      1. 归一化：消除特征值的量纲，具有可比性
      2. 将构建好的矩阵中的每个元素作为一个像素点，矩阵中的值作为像素的灰度（白的程度
   3. 卷积神经网络
      1. 输入层、卷积层、池化层、全连接层、输出层
         1. 卷积层是卷积神经网络中最重要的一部分？
         2. 池化层也称为采样层？
         3. 全连接层通常位于卷积神经网络模型的最后位置，作用是计算网络的最终输出结果
   4. 训练过程
      1. 卷积层的前向传播形式为 激活函数为 ReLU 函数（去除负数
      2. 反向传播算法和BP神经网络类似，代价函数
   5. 改进的卷积神经网络结构
4. 实验测试与结果分析

## 资料六：《卷积神经网络研究综述_周飞燕.pdf》

[大白话讲解卷积神经网络工作原理](https://www.bilibili.com/video/av35087157/?spm_id_from=333.788.videocard.0)

1. 引言
   1. 人工神经元网络 (Artificial Neural Network，ANN)是对生物神经网络的一种模拟和近似
   2. 反向传播网络 (Back Propagation Network，BP网络) **损失函数** => **通过训练自己得出卷积核**
   3. 逐层预训练 ”(layer-wise pre-training)
      1. 每次只无监督训练一层
      2. 将该层的训练结果作为其下一层的输入
      3. 有监督学习(BP算法)微调预训练好的网络
2. CNN概述
   1. 神经元：输入 权重 偏置 激励 输出
   2. 多层感知器：输入层、隐含层(一层或者多层)及输出层
   3. CNN：呈现出**图片结构**的数据
      1. 生物视觉感知
      2. 卷积层：通过`卷积核`提取出特征面，越高卷积层提取更高级的特征
      3. 池化层：降低特征面的分辨率
         1. 最大池化
         2. 平均池化
      4. 全连接层：用BP网络训练得到`权重`后，可导出分类结果
      5. `特征面`：根据实际应用进行设置
      6. 初始参数的确定（人为确定框架：找现成
         1. 卷积核数目、尺寸
         2. 池化大小、池化步长
         3. 全链接层神经元数量
3. 改进算法

## 资料七：基于深度置信网络的入侵检测模型

[深度信念神经网络DBN最通俗易懂的教程](https://blog.csdn.net/u013631121/article/details/76794829?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)

[Deep Learning（深度学习）学习笔记整理系列](https://blog.csdn.net/zouxy09/article/details/8775360)

1. 引言
2. 深度置信网络
   1. 由几个RBM和一层BP神经网络组成
   2. Restricting Boltzmann machine
      1. 一种两层随机神经网络
      2. 视觉层（v）
      3. 隐藏层（h）
      4. 能量函数
   3. BP神经网络
3. 入侵检测模型
4. 仿真与分析

## 资料八：基于深度学习的网络流量分类及异常检测方法研究_王伟

1. 绪论
2. 基础知识和相关工作
   1. 网络流量分类方法
      1. 一般的网络流量分类
         1. 基于端口识别的分类方法
            1. 通过传输层的端口号识别 65535个 查表法
         2. 基于深层包检测的分类方法
            1. 检查整个数据包的内容
            2. 匹配固定的字符、字符串模式：也叫signature指纹
            3. 无法处理加密流量、未知流量
         3. 基于统计的分类方法
            1. 提取数据特征-机器学习
            2. 仅需特征数据，可用于加密流量
         4. 基于行为的分类方法
            1. 使用主机通信的行为信息：协议、端口、数量
      2. 恶意流量分类
      3. 加密流量分类
   2. 网络流量异常检测方法
   3. 深度学习

## 资料九：[读懂反向传播算法（bp算法）](https://www.jianshu.com/p/74bb815f612e)

| 结构       | 神经元出    | 神经元入                                 | 神经元出 | 神经元入                                       |
| ---------- | ----------- | ---------------------------------------- | -------- | ---------------------------------------------- |
| 位置 层    | $l-1$       | $l$                                      | $l$      | $l+1$                                          |
| 位置 行    | $k$         | $j$                                      | $j$      | $i$                                            |
| 变量(向量) | $a_k^{l-1}$ | $a_k^{l-1}(A^{l-1})$                     | $a_j^{l}$ | $a_j^{l}(A^{l})$                               |
| 参数(向量) |             | $\omega_{jk}^{l}(W_{j}^{l})$ , $b_j^{l}$ |          | $\omega_{ij}^{l+1}(W_{i}^{l+1})$ , $b_i^{l+1}$ |

`正向运算过程：`

- 神经元变量输入：$a_k^{l-1}$

- 神经元参数输入：$\omega_{jk}^{l}$，$b_j^{l}$
  > $k$取值范围为上一层神经元数，$j$对于单一神经元取值唯一
- 神经元处理：

  - $z_j^{l} = W_{j\_}^{l}\cdot A^{l-1}+b_j^{l}$
    > $a_k^{l-1}$,$\omega_{jk}^{l}$ 排成$k$维列向量 $A^{l-1}$,$W_{j\_}^{l}$
    >
    > 输入值对$l$层的任意神经元$j$都相同，故用$A^{l-1}$ 统一表示
  - $a_j^{l} = \sigma(z_j^{l})$

- 神经元输出：$a_j^{l} = \sigma(z_j^{l})$

`反向处理过程：`

- 损失函数：$C(A^{L})$
  > L为总层数
- 定义：$\delta_{j}^{l} \equiv \frac{\partial C}{\partial z_j^{l}},\Delta^{l} \equiv \frac{\partial C}{\partial Z^{l}}$
  > 对$l$层$j$个神经元而言，中间变量$\delta_{j}^{l}$为**损失函数**对**未激活的输出值**的偏微分

- 计算：

  - 递推公式：$\delta_{j}^{l} =\frac{\partial C}{\partial Z^{l+1}}\cdot \frac{\partial Z^{l+1}}{\partial z_j^{l}} = \Delta^{l+1} \cdot W_{\_j}^{l+1} \cdot \sigma^{,}(z_j^{l})$
    > $z_i^{l+1},\delta_i^{l+1},\omega_{ij}^{l+1}$ 排成$j$维列向量 $Z^{l+1},\Delta^{l+1},W_{\_j}^{l+1}$
    >
    > $W_{\_j}^{l+1}$ 由所有从神经元$j$指出去的边的权重$\omega_{ij}^{l+1}$排成
    >
    > 与上文由所有指向神经元$j$的边的权重$\omega_{kj}^{l}$不同！！
  - 末值：$\delta_{j}^{L} =\frac{\partial C}{\partial A^{L}}\cdot \frac{\partial A^{L}}{\partial z_j^{L}}$
    > 特例为单输出值情况
    >
    > $\frac{\partial C}{\partial a^{L}}$ 取决于损失函数形式，一般为$\frac{1-y}{1-a^{L}}-\frac{y}{a^{L}}$
    >
    > $\frac{\partial a^{L}}{\partial z^{L}}$ 取决于激活函数形式，sigmoid时为$a^{L}\cdot(1-a^{L})$
    >
    > $\delta^{L} = a^{L} - y$

  - 用法：

    - $\frac{\partial C}{\partial \omega_{jk}^{l}} = \frac{\partial C}{\partial z_{j}^{l}} \cdot \frac{\partial z_{j}^{l}}{\partial \omega_{jk}^{l}} = \delta_{j}^{L} \cdot a_{k}^{l-1}$
  
    - $\frac{\partial C}{\partial b_{j}^{l}} = \frac{\partial C}{\partial z_{j}^{l}} \cdot \frac{\partial z_{j}^{l}}{\partial b_{j}^{l}} = \delta_{j}^{L}$

## 应用一：Tensorflow install

[Essential_documentation](https://tensorflow.google.cn/guide/)

1. 安装
   1. 网址 [TensorFlow](https://tensorflow.google.cn/)
   2. 软件需求：python, pip, virtualenv\conda
   3. 首次安装的版本可能并不契合网络资源所要求的tensorflow及其所需的python版本，因此可以使用conda创建虚拟环境选择合适的版本。
   4. tensorflow gpu 需要nvida显卡的支持，新款macbook搭载的是amd显卡，因此gpu版本安装问题无解
   5. 步骤
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

      - 通过pip包管理器
        - 从网址[package location](https://tensorflow.google.cn/install/pip#package-location)安装所需的tensorflow包

      ``` shell
      (venv) > pip install --ignore-installed --upgrade (packageURL)
      (venv) > pip install --upgrade tensorflow
      ```

        - 通过[清华源](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)安装

      ``` shell
      (venv) > pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.12.0
      ```

2. 使用
   1. 步骤
      - 激活环境（也可以通过anaconda图形界面打开终端），进入python

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

## 应用二：tensorflow.keras

[tensorflow-tutorials](https://tensorflow.google.cn/tutorials)

[Keras 快速搭建神经网络 (莫烦 Python 教程)](https://www.bilibili.com/video/av16910214)

1. QuickStart
2. Keras
3. Load data
4. Estimator
5. Customization
6. Distributed training
7. Images
   1. CNN
      - [tf.keras.layers.Conv2D](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2D)
        - filters 输出空间future map的维数/层数/高度、过滤器（卷积核）的数量
        - kernel_size 卷积核大小
        - strides 步长
        - activation 激活函数
        - use_bias 是否使用偏置向量b，默认为true
        - 作为第一层需要额外提供，例如input_shape=(128, 128, 3) 用于中的128x128 RGB图片
      - [tf.keras.layers.MaxPool2D](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/MaxPool2D)
        - pool_size (2, 2)将在两个空间维度上将输入减半
        - strides 步长 默认为上值
      - 输出形状： 具有形状的4D张量： (samples, filters, new_rows, new_cols)

      ```py
      model = models.Sequential()
      model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
      #卷积核长与宽自己定，层数由输入图片层数决定，偏置bias b_i[1,1,1]，然后输出一张图 w_i[3,3,3]
      #输出次数自定 i[32]
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation='relu'))
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation='relu'))
      model.add(layers.Flatten()) # input 4*4*64 output 1024
      model.add(layers.Dense(64, activation='relu'))
      model.add(layers.Dense(10))
      model.summary()

      Model: "sequential_1"
      _________________________________________________________________
      Layer (type)                 Output Shape              Param #
      =================================================================
      conv2d_3 (Conv2D)            (None, 30, 30, 32)        896 = (3*3*3+1)*32
      _________________________________________________________________
      max_pooling2d_2 (MaxPooling2 (None, 15, 15, 32)        0
      _________________________________________________________________
      conv2d_4 (Conv2D)            (None, 13, 13, 64)        18496 = (3*3*32+1)*64
      _________________________________________________________________
      max_pooling2d_3 (MaxPooling2 (None, 6, 6, 64)          0
      _________________________________________________________________
      conv2d_5 (Conv2D)            (None, 4, 4, 64)          36928 = (3*3*64+1)*64
      _________________________________________________________________
      flatten_1 (Flatten)          (None, 1024)              0
      _________________________________________________________________
      dense_2 (Dense)              (None, 64)                65600
      _________________________________________________________________
      dense_3 (Dense)              (None, 10)                650
      =================================================================
      Total params: 122,570
      Trainable params: 122,570
      Non-trainable params: 0
      _________________________________________________________________
      ```

8. Text
9. Structure data
10. Generative
    1. DCGan
       1. [batch nomolization](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/BatchNormalization)
          - [【 深度学习李宏毅 】 Batch Normalization （中文）](https://www.bilibili.com/video/av16540598/)
          - normalization 意义：损失函数对范围大数值对应参数w更敏感，需要不同对学习速率/同化到相同范围
            - 后层的变化总是落后与前层变化
            - 保证每个layer输出相对固定，
            - 问题？但每个layer输出的mean和variant是不断变化的
          - batch normalization：可以调大rate加快训练速度，解决vanishing gradient
            - 可以在激活前后应用（推荐先normalization后activation：让值落在零的附近再激活效果较好）
            - batch 一串 把多个样本
            - 对每个layer的一串输出z求出z均值和z方差，然后根据均值和方差进行normalization
            - 如何bp？z均值和z方差也视为变量，而normalization后可以进行放缩和平移（常量）
            - training时的z均值和z方差在不断变化，在考虑正确率作为权重

   ```py
   #generator
   def make_generator_model():
      model = tf.keras.Sequential()
      model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Reshape((7, 7, 256)))
      assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

      model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
      # b=0: not use bias
      assert model.output_shape == (None, 7, 7, 128)
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
      assert model.output_shape == (None, 14, 14, 64)
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
      assert model.output_shape == (None, 28, 28, 1)

      return model

   generator = make_generator_model()
   _________________________________________________________________
   Layer (type)                 Output Shape              Param #
   =================================================================
   dense (Dense)                (None, 12544)             1254400 = 7*7*256*100
   _________________________________________________________________
   batch_normalization (BatchNo (None, 12544)             50176 = 12544*4????
   _________________________________________________________________
   leaky_re_lu (LeakyReLU)      (None, 12544)             0
   _________________________________________________________________
   reshape (Reshape)            (None, 7, 7, 256)         0
   _________________________________________________________________
   conv2d_transpose (Conv2DTran (None, 7, 7, 128)         819200 = (5*5*256+0)*128
   _________________________________________________________________
   batch_normalization_1 (Batch (None, 7, 7, 128)         512 = 128*4
   _________________________________________________________________
   leaky_re_lu_1 (LeakyReLU)    (None, 7, 7, 128)         0
   _________________________________________________________________
   conv2d_transpose_1 (Conv2DTr (None, 14, 14, 64)        204800 = (5*5*128+0)*64
   _________________________________________________________________
   batch_normalization_2 (Batch (None, 14, 14, 64)        256 = 64*4
   _________________________________________________________________
   leaky_re_lu_2 (LeakyReLU)    (None, 14, 14, 64)        0
   _________________________________________________________________
   conv2d_transpose_2 (Conv2DTr (None, 28, 28, 1)         1600 = (5*5*64+0)*1
   =================================================================
   Total params: 2,330,944
   Trainable params: 2,305,472
   Non-trainable params: 25,472
   _________________________________________________________________



   #discriminator
   def make_discriminator_model():
      model = tf.keras.Sequential()
      model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                       input_shape=[28, 28, 1]))
      model.add(layers.LeakyReLU())
      # [tf.keras.layers.LeakyReLU](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/LeakyReLU)
      model.add(layers.Dropout(0.3))
      # [tf.keras.layers.Dropout](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Dropout) 将输入单元的一部分随机设置为0，这有助于防止过拟合。

      model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
      model.add(layers.LeakyReLU())
      model.add(layers.Dropout(0.3))

      model.add(layers.Flatten())
      model.add(layers.Dense(1))

      return model

   discriminator = make_discriminator_model()
   _________________________________________________________________
   Layer (type)                 Output Shape              Param #
   =================================================================
   conv2d (Conv2D)              (None, 14, 14, 64)        1664 = (5*5*1+1)*64
   _________________________________________________________________
   leaky_re_lu_3 (LeakyReLU)    (None, 14, 14, 64)        0
   _________________________________________________________________
   dropout (Dropout)            (None, 14, 14, 64)        0
   _________________________________________________________________
   conv2d_1 (Conv2D)            (None, 7, 7, 128)         204928 = (5*5*64+1)*128
   _________________________________________________________________
   leaky_re_lu_4 (LeakyReLU)    (None, 7, 7, 128)         0
   _________________________________________________________________
   dropout_1 (Dropout)          (None, 7, 7, 128)         0
   _________________________________________________________________
   flatten (Flatten)            (None, 6272)              0
   _________________________________________________________________
   dense_1 (Dense)              (None, 1)                 6273
   =================================================================
   Total params: 212,865
   Trainable params: 212,865
   Non-trainable params: 0
   _________________________________________________________________



   # Generator loss
   def generator_loss(fake_output):
      return cross_entropy(tf.ones_like(fake_output), fake_output)
   # Discriminator loss
   def discriminator_loss(real_output, fake_output):
      real_loss = cross_entropy(tf.ones_like(real_output), real_output)
      fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
      total_loss = real_loss + fake_loss
      return total_loss
   # discriminator and the generator optimizers ?????
   # [tf.keras.optimizers.Adam](https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/Adam?hl=en)
   generator_optimizer = tf.keras.optimizers.Adam(1e-4) #实现Adam算法的优化器。学习率
   discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)



   # training loop!!!
   def train_step(images):
      noise = tf.random.normal([BATCH_SIZE, noise_dim]) #根据batch大小生成随机数据

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
         generated_images = generator(noise, training=True) #生成器，输出图片数据(None, 28, 28, 1)

         real_output = discriminator(images, training=True) #判别器，输出判别数据(None, 1)
         fake_output = discriminator(generated_images, training=True) #判别器，输出判别数据(None, 1)

         gen_loss = generator_loss(fake_output)
         disc_loss = discriminator_loss(real_output, fake_output)

      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables) # 调用GradientTape.gradient方法，就会释放GradientTape拥有的资源
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)) #input:List of (gradient, variable) pairs
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

   def train(dataset, epochs):
      for epoch in range(epochs):
         start = time.time()

         for image_batch in dataset: #BATCH_SIZE = 256
            train_step(image_batch) #image_batch.shape=(256, 28, 28, 1)

         # Produce images for the GIF as we go
         display.clear_output(wait=True)
         generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

         # Save the model every 15 epochs
         if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

         print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

      # Generate after the final epoch
      display.clear_output(wait=True)
      generate_and_save_images(generator,
                                 epochs,
                                 seed)

   train(train_dataset, EPOCHS) #EPOCHS = 50 迭代五十次

   ```

