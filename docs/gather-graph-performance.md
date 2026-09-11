# Gather GraphOptimize 性能测试

## 主性能回归：PR #1027 原始负载

主入口是 `third_party/ascend/unittest/pytest_ut/bench_gather_pr1027.py`，kernel 位于
同目录的 `gather_pr1027_kernel.py`。来源是
[PR #1027 的原始附件](https://github.com/user-attachments/files/30009694/test_gather_optimization.py)。

此入口保留原附件的全部 14 组 shape、ND kernel、FP32 数据/i32 索引、最后一维 gather、
`CORES=40` 和 `UB_SIZE=192*1024`，以及原有 ROW_STEP 的向上取整公式。没有改成
二维 flatten、补齐宽度或重新选择 tiling。包括 `K=3076`、`ROW_STEP=3` 和不能整除
的 `ROW_BLK=1639`。kernel 的地址表达式、load/store mask、other=0、N_ROWS 等
constexpr 参数也保持原样。主机测试通过 AST 摘要检查 shape 和 kernel 与附件一致。

外围驱动加入规则掩码 511/1023 对照：两版本使用同一份源、索引和输出缓冲区，只有
规则掩码不同。每个版本计时前单独与 torch.gather 参考比较（零容差），并先将输出
置为 NaN，避免漏写被上一版本的结果掩盖。参考计算需要的 i64 转换仅用于
torch.gather，实际被测 kernel 仍接收原来的 i32 索引。torch.gather 不参与计时。

在已安装当前代码的服务器环境中，从仓库根目录执行：

```bash
unset ASCEND_LAUNCH_BLOCKING PYTORCH_NO_NPU_MEMORY_CACHING
unset TRITON_ALWAYS_COMPILE TRITON_ENABLE_SANITIZER
BENCH=third_party/ascend/unittest/pytest_ut/bench_gather_pr1027.py

# 无需 NPU：列出原始形状及 tiling；ID 为 1～14。
python "$BENCH" --list-cases

# 原始第 1 组，shape 和 tiling 不缩小，只减少采样确认流程。
python "$BENCH" --device 0 --case-ids 1 \
  --rounds 1 --warmup 3 --repeat 10 \
  --output-dir gather-v3-perf/pr1027-smoke

# 正式主回归：全部 14 组，所有版本使用同一种 profiler 计时。
python "$BENCH" --device 0 --rounds 3 --warmup 5 --repeat 30 \
  --output-dir gather-v3-perf/pr1027-full
```

脚本直接使用仓库的 `do_bench_npu`，无需设置 TRITON_BENCH_METHOD。默认不主动清
L2；若要比较清缓存条件，单独运行并添加 `--clear-l2-cache`，使用不同输出目录。
warmup/repeat 是每轮调用次数；汇总耗时为各轮平均设备 kernel 耗时的中位数，统一
转换为微秒。采样顺序在 511/1023 间交替。编译、初始化、精度检查均在计时之外。
每个输出目录使用新编译缓存并关闭 sanitizer，目录必须不存在。

每次只保留一个配置的张量，完成后释放其引用。原始规模较大，单组源、索引和输出
合计约 384～2304 MiB，正确性参考和临时张量还会额外占用内存。脚本保留这些规模。

| ID | 源 shape | 索引/输出 shape | ROW_STEP | 每 program 循环次数 |
| --- | --- | --- | --- | --- |
| 1 | (65536,4096) | (65536,2048) | 2 | 820 |
| 2 | (65536,2048) | (65536,3076) | 2 | 820 |
| 3 | (65536,1024) | (65536,4096) | 1 | 1639 |
| 4 | (65536,64,64) | (65536,64,32) | 2 | 820 |
| 5 | (65536,64,32) | (65536,64,32) | 2 | 820 |
| 6 | (65536,16,64) | (65536,16,128) | 2 | 820 |
| 7 | (65536,16,16,16) | (65536,16,16,8) | 2 | 820 |
| 8 | (65536,16,8,16) | (65536,16,8,16) | 2 | 820 |
| 9 | (65536,4,8,32) | (65536,4,8,64) | 2 | 820 |
| 10 | (65536,4,8,8,8) | (65536,4,8,8,4) | 3 | 547 |
| 11 | (65536,4,4,2,16) | (65536,4,4,2,16) | 8 | 205 |
| 12 | (65536,8,2,2,8) | (65536,8,2,2,32) | 4 | 410 |
| 13 | (65536,2,2,1,64) | (65536,2,2,1,256) | 4 | 410 |
| 14 | (65536,32,2,2,2) | (65536,32,2,2,8) | 4 | 410 |

所有配置 grid=(40,)，ROW_BLK=1639。若某个工具链不能编译其中的非 2 幂配置，记录
真实编译错误，使用 `--case-ids` 单独排查；不要修改 shape/ROW_STEP 后仍视为原始回归。

主回归输出：

- `run.json`：全部选中配置、版本、设备、编译器、源码位置、提交和环境变量。
- `summary.csv`：每组 shape/tiling、`status`、`rewritten`、off/on 微秒耗时及各轮范围。
  `time_ratio=off_us/on_us` 始终记录；只有实际命中时才填写 `speedup`。
- `samples.csv`：各轮原始平均耗时，逐条写入。
- `case_XX/case.json`、`compile.json`：该组参数、命中状态及实际 kernel 名称。
- `case_XX/mask511.ttir`、`mask1023.ttir`：两份编译结果。
- `case_XX/profile_r*_mask*/`：原始 profiler 报告；按实际 kernel 名称过滤并检查采样数。

关闭规则却出现 tt.gather 会立即报错。开启后未命中时记录 `not_rewritten`，保留该组
原始参数和耗时，`speedup` 留空，继续运行其他配置。当前收益/UB 门槛可能拒绝原附件
中的部分配置，不能通过改 shape 掩盖。需要严格验收全部命中时加 `--require-rewrite`，
它在完成所选配置后因任何未命中返回非零退出码。

编译、精度或 profiler 错误会写入该组 `error.txt` 和汇总的 `error` 状态，然后停止；
已完成的结果保留。避免设备异常后继续执行产生连锁错误。用新目录和 `--case-ids`
可以单独重跑某组。

主机回归入口（无需 NPU 执行）：

```bash
python -m pytest -q third_party/ascend/unittest/pytest_ut/test_bench_gather_pr1027.py
```

## 二维诊断微基准

`bench_gather_graph_optimize.py` 保留为补充诊断，下面的配置不代替上述原始 ND 主回归。

现有 `test_gather_graph_optimize.py` 是正确性回归，不能用 pytest 总耗时评估性能。
`test_gather_simd.py` 和 `tutorials/10-gather-2d-simd.py` 已经显式使用
`tl.gather`，不能直接测量间接 load 自动改写的收益。

`third_party/ascend/unittest/pytest_ut/bench_gather_graph_optimize.py` 复用已验证的
`indirect_rows_kernel`，使用仓库的 `triton.backends.ascend.testing.do_bench_npu`
采集设备上的 kernel 耗时。两份编译仅改变规则掩码：511 关闭 gather；1023 启用
gather。其他规则、输入、grid 和 tiling 一致；索引为 i32。

共享 kernel 显式计算 `offsets = base + indices` 后再做 `src_ptr + offsets`，
以符合 v3 规则的直接指针匹配要求。旧写法 `src_ptr + base + indices` 会生成嵌套
指针加法；动态 i32 偏移因溢出语义不同，前序 pass 不保证将其合并，可能导致规则
不命中。更新性能入口时也必须同步 `test_gather_graph_optimize.py`。

## 运行

在已构建并安装当前代码的服务器环境中，从仓库根目录执行。需要 torch、torch_npu、
pytest（复用测试模块）、numpy、pandas，以及可工作的 torch_npu profiler。
不要用 pytest 或多进程运行此脚本；请选择空闲 NPU。

先清除此前排查时可能设置的同步/强制编译选项。脚本还会强制关闭 sanitizer 和
ALWAYS_COMPILE，并在每次输出目录下创建独立编译缓存，避免复用插桩二进制。
清理缓存、编译、分配、CPU 参考计算和精度检查均不作为被测 kernel 的耗时。

```bash
unset ASCEND_LAUNCH_BLOCKING PYTORCH_NO_NPU_MEMORY_CACHING
unset TRITON_ALWAYS_COMPILE TRITON_ENABLE_SANITIZER
BENCH=third_party/ascend/unittest/pytest_ut/bench_gather_graph_optimize.py

# 2 个点，确认性能采集和 gather 改写均正常；本轮不用于最终性能结论。
python "$BENCH" --device 0 --cases full tail --require-rewrite \
  --rounds 1 --warmup 3 --repeat 10 \
  --output-dir gather-v3-perf/smoke

# 48 个点：4 种源宽度 × 3 种索引宽度 × 2 种类型 × 2 种边界。
python "$BENCH" --device 0 \
  --widths 16 64 256 1024 --index-widths 4 8 32 \
  --dtypes float32 float16 --cases full tail \
  --rounds 3 --warmup 5 --repeat 30 \
  --output-dir gather-v3-perf/sweep

# 8 个点：每个 tile 都遇到合法的负偏移/下一行偏移，测试运行时 fallback 开销。
python "$BENCH" --device 0 \
  --widths 16 256 --index-widths 8 32 --cases negative upper_bound \
  --rounds 3 --output-dir gather-v3-perf/fallback

# 使用仓库提供的清 L2 操作，独立采集另一种缓存条件；不要混合两组统计。
python "$BENCH" --device 0 \
  --widths 16 64 256 1024 --index-widths 4 8 32 \
  --dtypes float32 float16 --cases full tail --clear-l2-cache \
  --rounds 3 --output-dir gather-v3-perf/sweep-clear-l2
```

输出目录必须是新目录，重跑时换名字。默认 rows=4096、row-block=128、row-step=4：
grid 为 32，每个 program 处理 128 行，分 32 次循环；tail 只有 4095 行有效数据。
这是固定 tiling 的微基准，不声称适配所有 NPU 的最优分核。需要比较 tiling 时，
单独改变 `--rows`、`--row-block`、`--row-step`，确保开关双方仍使用相同配置。
rows 必须能被 row-block 整除，row-block 必须能被 row-step 整除；宽度和 row-step 限制为
正的 2 的幂。本入口覆盖二维 kernel，不复现历史的 rank 2–5 非对齐宽度全量数据集。

## 计时与结果解释

- 每个点先编译、同步并检查两份输出与 CPU 指针索引参考完全相同。full/tail 的活跃
  索引位于 `[0, WIDTH)`；negative/upper_bound 则在每行放入一个越出 gather 轴范围
  但物理地址合法的索引。后两者若生成了 gather，运行时每个 tile 都走 fallback。
- 默认不主动清 L2，重复使用同一份数据。`--clear-l2-cache` 使用后端的清缓存操作；
  它模拟另一种缓存条件，不能保证所有硬件和所有访问都是完全冷缓存。
- 每轮每个版本使用独立 profiler 会话，保留全部原始报告；通过编译元数据中的
  `kernel_name` 过滤记录，并校验采样数，避免计入其他 NPU 算子。
- `--warmup` 和 `--repeat` 的单位都是调用次数。do_bench_npu 返回测量调用的平均
  kernel 耗时（毫秒），脚本转换为微秒；汇总为多轮均值的中位数。
- 脚本交替两版本的测量顺序。默认 3 轮，重点波动/退化点可以提高到 5 或 7 轮。
  这是设备 kernel 时间，不是 Python 调用或端到端应用延迟。
- 不设置 `--require-rewrite` 时，收益或 UB 门槛拒绝的形状也会保留。
  `rewritten=False` 表示开关没有带来该规则的改写，不能算作 gather 加速结果。
  此开关不会绕过门槛，无法单独判断被拒绝形状是否其实有收益。

输出：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 提交、工作区状态、设备、版本、Python 后端/原生扩展/kernel 源码路径、编译器路径、关键环境变量和参数 |
| `summary.csv` | 每个点的 off/on 微秒耗时、加速比、各轮均值的最小/最大值、改写命中 |
| `samples.csv` | 每轮每个版本的耗时与 kernel 名称，逐条落盘 |
| `<point>/mask511.ttir`、`mask1023.ttir` | 编译后 TTIR，用于核对是否触发 gather |
| `<point>/profile_r*_mask*/` | 原始 profiler 报告，包括 `kernel_details.csv` |
| `cache/` | 本次独立的 Triton 编译缓存 |

`speedup = off_us / on_us`，大于 1 为加速，小于 1 为退化。先查看
`rewritten=True` 的 full/tail 结果，再独立看 fallback。不要把未命中的点或不同
缓存条件合并成一个平均加速比。可将重复测量后仍慢于 5% 的点列为优先分析对象；
5% 是筛选建议，不是当前优化器的收益保证。

## 后续扩展

先完成上述 sweep，再对有代表性的形状改变 row-step 和循环次数，检查收益阈值随
迭代次数变化的行为；另测 bfloat16。靠近 UB 上限的 tile、包含其他中间张量或多个
gather 的融合 kernel，以及三至五维输入需要专门用例，不能由本微基准代替。

比较 v2/v3 时，应在两个已正确安装的构建环境中使用同一 kernel、输入、tiling 和
计时流程，并保存各自的 run.json。应先比较两个版本都正确的 full 输入；v2 对尾块
或 fallback 的错误输出不能作为有效性能基线。当前命令首先回答 v3 中这条规则的
净收益，没有承诺复现旧 PR 的峰值加速比。
