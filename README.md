# solo-bookit

单资源预约排程命令行工具，数据保存在工作目录下的本地文件中。

- 仅使用 Python 标准库，不联网。
- 入口：`python bookit.py <子命令>`
- 时间一律为本地时间的 ISO 8601 字符串；同一资源上不允许存在时间重叠的预约。

## 测试

    python -m unittest discover
