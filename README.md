# mock-vllm

在**没有 GPU** 的机器上运行 vLLM 的 host（CPU）侧代码路径。

它保留 vLLM 的 Python 调度、输入前处理、采样和模型层循环，只把设备下发
（ixformer kernel、分布式通信 pybind、CUDA runtime、cudagraph 回放）替换成
C 级空实现。用来在不占卡、不出 kernel 的前提下度量并对比 host 侧耗时，
例如 litehost 的 H/G 指标。

> 只用于 host 侧性能分析与 CPU-only 环境验证，**不产生正确的推理结果**。

## 它做了什么

| 保留 | 替换为空实现 |
|---|---|
| scheduler / 请求调度、tokenizer、输入组装 | ixformer 算子下发（device leaf） |
| 模型 forward 的 Python 循环（真实层结构） | 分布式集合通信（单 rank 桩） |
| sampler、logits 处理、输出回写 | CUDA runtime / Stream / Event 同步 |
| cudagraph 的调用结构与 scope 划分 | 图回放（`CUDAGraph.replay`） |
| profiling / tracing 的 Python 帧 | 显存分配与实际 kernel 执行 |

### 激活方式

`sitecustomize.py` 挂在 `PYTHONPATH` 上，Python 启动时自动 import，
由它 import `mock_gpu.autoload`。`autoload` 只在**检测不到设备**时安装补丁：

- `CUDA_VISIBLE_DEVICES=-1`，或
- `/dev/iluvatar0` 不存在

然后依次做三件事：

1. **平台层替换**（`vllm.platforms.current_platform`）：device 改成 `cpu`、
   关闭 compilation、替换 device communicator / 显存查询 / graph pool。
2. **worker 补丁**（`mock_gpu/worker_patch.py`）：假 Stream/Event/sync、
   设备叶子算子的空替身、权重加载窗口内的拷贝替身、帧名对齐。
3. **runner 替换**（`mock_gpu/runner.py` 的 `MockGPUModelRunner`）：
   在 CPU 上构造并加载真实模型结构，采样与设备相关步骤换成替身。

`fake_graph.py` 提供的假 `CUDAGraph` 把 `replay` 编译成与
`torch/cuda/graphs.py:141` 同名同行的帧，保证 trace 里调用栈位置一致。

有卡机器上不加这两个条件时 mock 不会启动，可以安全放置。

## 兼容性

- Linux x86_64，Python 3.11，装有 Iluvatar 版 vLLM / torch / ixformer 的 venv。
- 补丁按**名字和文件行号**打（vLLM / torch / ixformer 内部），因此绑定具体版本。
  已验证组合：vLLM 0.23.0 / torch 2.10.0 / triton 3.2.0。
- 更换版本后先跑一次冒烟，确认 `[mock_gpu] activated` 和每个 worker 的
  `[mock_gpu] worker patched` 都出现。

## 快速开始

```bash
git clone <this repo> /path/to/mock-vllm
cd /path/to/mock-vllm

# 可选：编译 litehost 量 H/G 用的符号垫片
gcc -shared -fPIC -o mock_ix.so mock_ix.c

# 用 PYTHONPATH 激活，不装包、不改 vllm 源码
PYTHONPATH=/path/to/mock-vllm \
CUDA_VISIBLE_DEVICES=-1 \
  /path/to/venv/bin/python -m vllm.entrypoints.openai.api_server ...
```

启动日志应出现：

```
[mock_gpu] activated: no Iluvatar device visible (id=xxxxxxxxxx)
[mock_gpu] worker patched
```

去掉 `PYTHONPATH` 即完全卸载。

> `sitecustomize.py` 是兜底入口。如果 venv 里已经有别的 `sitecustomize`，
> 改用 `.pth` 注入：
> ```bash
> printf '%s\nimport mock_gpu.autoload\n' "$PWD" \
>   > $VENV/lib/python3.11/site-packages/zz_mock_gpu.pth
> ```
> 删除该文件即卸载。

## 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `MOCK_THREADS` | `1` | torch intra-op 线程数，对齐 GPU worker 的单线程设置 |
| `MOCK_PREFILL` | `1` | `0` = 用假模型跳过真实层循环，只测框架路径 |
| `MOCK_EAGER` | `0` | `1` = 关闭 cudagraph，全部 eager 执行 |
| `MOCK_SAMPLE_CAST` | `1` | `0` = 不做采样器的 fp32 大张量转换（对照组） |
| `MOCK_IX_LIB` | `<repo>/mock_ix.so` | litehost 符号垫片路径 |
| `LIGHTHOST_LIB` / `LIGHTHOST_SESSION` | 由 litehost 设置 | 存在时同步/图下发走 `mock_ix.so`，H/G 测量才有效 |

## 版本标识

启动日志里的 `id=xxxxxxxxxx` 是 `mock_gpu/*.py` 的内容哈希，自动计算、
无需手工维护。**H 数据只与同一 id 的结果可比**；结果目录
（`config.txt` 或 `logs/server.log`）里应记录该 id，冻结基线时同时记录
`git rev-parse HEAD`。

## 与 base 的对齐程度

Qwen3-32B-W8A8，TP=4，输入/输出 128/128，`execute_model` 窗口内的 Python 帧名
逐项比对（rank0，中段 step）：

| | base 帧名/调用 | mock 帧名/调用 | 共同帧名 |
|---|---|---|---|
| prefill | 1273 / 30584 | 1202 / 30282 | 1196 |
| decode | 249 / 727 | 207 / 573 | 204 |

mock 独有帧只有 3 类：设备叶子替换点 `_noop`（与 base 设备叶子一一对应）、
`linear_ex` 的 `<lambda>`、采样器 fp32 转换的 `to`；prefill 另有 CPU inductor
的 mask kernel 3 帧。base 独有的都是设备叶子、CUDA runtime/Event、
`torch.cuda` 懒初始化链、triton/inductor 的启动机制。

### 已知偏差（做 A/B 时按常量看待）

- **设备提交成本整体不在 mock 里**：base 每步的 triton JIT 启动链（slot-mapping
  内核）约 100–200 µs（profile 口径），prefill 的上千次设备叶子调用合计约
  22 ms；mock 对应位置是空替身（µs 级）。因此 mock 的 H 天然远低于 base，
  **只能比相对差值**，且“减少下发次数/启动开销”类优化在 mock 上看不出来。
- prefill 的 causal-mask kernel 在 mock 里编成 CPU kernel 运行
  （约 13 µs/step，结果无人读取）。
- trace 时长不能直接比：base 的 `execute_model` 包含 device 等待，
  profiler 对每步数万帧也有放大。

## 目录

```
.
├── mock_gpu/
│   ├── autoload.py      # 入口：无卡时打补丁，打印版本 id
│   ├── worker_patch.py  # 假 Stream/Event/sync、设备叶子替身、加载窗口替身、帧名对齐
│   ├── runner.py        # MockGPUModelRunner：CPU 上构造并加载真实模型结构
│   ├── fake_graph.py    # 假 CUDAGraph（replay 帧名对齐 torch/cuda/graphs.py:141）
│   ├── dispatch.py      # 阶段计数与审计
│   ├── counters.py      # 计数器
│   └── comm.py          # 单 rank 通信桩
├── sitecustomize.py     # PYTHONPATH 自动激活入口
├── mock_ix.c            # litehost 符号垫片源码（同步/图下发空实现）
├── mock_ixrt.c          # 可选：IXRT 空实现源码，mock_gpu 不引用
└── cases9.txt           # 9 case 矩阵示例（input/output 长度组合）
```

## 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| 日志没有 `[mock_gpu] activated` | 机器上仍可见设备：设置 `CUDA_VISIBLE_DEVICES=-1`；或 `PYTHONPATH` 未指向本目录 |
| `[mock_gpu] skipped: ...` | vLLM 内部符号与补丁不匹配，版本变了，按新版重新对齐 |
| `[mock_gpu] sitecustomize skipped: ...` | venv 里已有 `sitecustomize` 冲突，改用 `.pth` 注入 |
| torch 报 CUDA 错误 | 未设置 `CUDA_VISIBLE_DEVICES=-1` |
| trace 里多出/缺少 Python 帧 | 两份结果的 mock id 不同，或 base 版本不一致 |

## 许可

未附带许可证文件。如需开源授权，请自行添加（例如 MIT / Apache-2.0）。
