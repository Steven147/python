# Jupyter notebook

[Jupyter Notebook介绍、安装及使用教程-jianshu](https://www.jianshu.com/p/91365f343585)

> Jupyter Notebook是基于网页的用于交互计算的应用程序。其可被应用于全过程计算：开发、文档编写、运行代码和展示结果。

## 基本使用

* 启动
  * 终端启动

  ```shell
  > jupyter notebook
  ```

  * 修改默认启动路径到/Users/lsq/Documents/GitHub/python
  
  ```shell
  > jupyter notebook --generate-config
  Writing default config to: /Users/lsq/.jupyter/jupyter_notebook_config.py
  > vim /Users/lsq/.jupyter/jupyter_notebook_config.py
  ```
  
    vim下直接输入```/c.NotebookApp.notebook_dir```进行搜索，回车直接定位，**去掉注释符号**，对引号中内容进行修改，然后```:wq```写入退出。

  