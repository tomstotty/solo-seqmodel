# solo-seqmodel

从零实现的序列建模库，仅用 Python 标准库、不联网。

- 入口：`python seqmodel.py <子命令>`
- 训练与采样都必须完全确定：相同种子与输入产生逐字节相同的输出。
- 语料来自仓库内的本地文本文件，不下载任何外部资源。

## 测试

    python -m unittest discover
