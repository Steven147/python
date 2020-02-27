# 日语五十音学习软件

## 下载

### 途径一

- 类Unix系统[下载](https://github.com/Steven147/python/raw/master/%E6%97%A5%E8%AF%AD%E4%BA%94%E5%8D%81%E9%9F%B3%E5%AD%A6%E4%B9%A0/aiueo)（适用于Linux、macOS）

- Windows系统[下载](https://github.com/Steven147/python/raw/master/%E6%97%A5%E8%AF%AD%E4%BA%94%E5%8D%81%E9%9F%B3%E5%AD%A6%E4%B9%A0/aiueo.exe)

## 功能

### ***核心思想：随机产生五十音进行辨识***

1. 按键一(NEW!)：生成一个假名的罗马字表示

    ![](2020-02-2813514.gif)

2. 按键二(Answer?)：显示对应的平假名、片假名

    ![](2020-02-2813908.gif)

3. 按键三(Testing/Learning)：切换测试/学习状态
   1. 测试状态下默认隐藏答案
   2. 学习状态下会始终显示答案

      ![](2020-02-2814230.gif)

## 实现（以下是我的絮絮叨叨

~~下载途径二（硬核~~

1. 下载完整的repository文件，建议采用zip格式
2. 安装python
   - [python官方网站下载链接](https://www.python.org/downloads/)，按照步骤执行

   - 检查按照情况

      ```bash
      > python3 --version #检验是否安装python
      Python 3.7.2 #成功安装则返回版本
      ```

   - 在系统**命令行界面/终端**执行下列语句

      ```shell
      > python3 /.../aiueo.py #python3，空格，加上程序文件的路径（可将程序拖入命令行生成路径
      ```
具体实现步骤

- 已实现功能
  - 随机生成罗马音组合
  - 显示罗马音对应平假名、片假名
  - 开启、关闭始终显示
- ~~待实现功能~~
  - ~~平假名、片假名字源信息显示~~

1. 五十音数据收集
2. 程序运行逻辑梳理，函数编写
   1. 按键一(NEW!)：随机生成 a k s t n h m y l w 与 a e u e o 组合
       - 特殊情况1: 在遇到没有对应名的音，则重新选择
       - 特殊情况2：ti -> chi
       - 特殊情况3：tu -> tsu
   2. 按键二(Answer?)：显示对应的平假名、片假名
   3. 按键三：开启始终显示/关闭始终显示
3. 交互实现
