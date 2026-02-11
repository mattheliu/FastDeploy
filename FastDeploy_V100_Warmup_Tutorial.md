# FastDeploy V100 热身打卡教程

## 🚀 活动说明

本教程是 [FastDeploy 热身打卡活动](https://github.com/PaddlePaddle/FastDeploy/issues/6225) 的 **V100 专用版本**，基于 [PR #6306](https://github.com/PaddlePaddle/FastDeploy/pull/6306) 的 V100 (SM70) 支持功能。

> **锁定版本**：本教程基于 commit `48adbc40fc29d0cd660311d141eff0ca48f037d2`
>
> **开发状态**：V100 支持功能正在持续开发中，欢迎通过本教程体验编译流程并反馈问题！

---

## 📋 V100 与 A100 的主要区别

| 特性 | V100 (SM70) | A100 (SM80) |
|------|-------------|-------------|
| BF16 | ✅ 支持（Tensor Core） | ✅ 原生支持 |
| FP8 | ❌ 不支持 | ⚠️ 需 SM89+ |
| cp.async | ❌ 不支持 | ✅ 支持 |
| APPEND_ATTN | ❌ 跳过编译 | ✅ 支持 |
| MLA_ATTN | ❌ 跳过编译 | ⚠️ 需 SM90+ |
| FP8 GEMM | ❌ 跳过编译 | ⚠️ 需 SM89+ |

### V100 跳过编译的算子

由于硬件限制，以下算子在 V100 上**不会编译**（通过 `setup_ops.py` 中的 `cc >= 80/89/90` 条件跳过）：

| 类别 | 跳过的算子/功能 | 原因 |
|------|----------------|------|
| Attention | `append_attention`, `multi_head_latent_attention` | 需要 cp.async/ldmatrix (SM80+) |
| MOE | `gptq_marlin_repack`, `winx_unzip` | 需要 SM80+ |
| FP8 量化 | `fp8_gemm_*`, `per_token_quant`, `fused_hadamard_quant_fp8` 等 | 需要 FP8 硬件 (SM89+) |
| Hopper 优化 | `mla_attn`, `flash_mask_attn`, `moba_attn`, `machete`, `w4afp8_gemm` | 需要 SM90+ |

> **注意**：V100 兼容性支持正在 PR #6306 中持续开发和完善中。您可以先通过本文档锁定的 commit 体验完整的编译流程。欢迎大家关注 PR 进展、提交反馈或参与开发！

---

## 🧰 准备环境

### 1. 硬件要求

- **NVIDIA V100 GPU** (SM70 架构)
- 推荐内存：>=32GB
- CUDA 11.x

### 2. 安装 PaddlePaddle

```bash
# V100 使用 CUDA 11.8 版本
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

### 3. 克隆 FastDeploy 源码

```bash
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy

# 切换到 V100 支持的 PR 分支（锁定 commit）
git fetch origin pull/6306/head:pr-6306
git checkout pr-6306
git reset --hard 48adbc40fc29d0cd660311d141eff0ca48f037d2

# 确认 commit
git log --oneline -1
# 预期输出: 48adbc40f Merge branch 'develop' into fastdeploy_v100
```

---

## 🔧 编译打卡流程

> **重要**：V100 编译时 MAX_JOBS 建议设置为 **8**，过高会导致 OOM 被 Kill。
> 所有关键步骤需加 `time` 记录耗时，并截图保存。

### Step 1：执行 FastDeploy 编译与打包

```bash
# 参数说明
# 第1个参数: 是否构建 wheel（1=构建，0=仅编译）
# 第2个参数: Python 解释器
# 第3个参数: 是否编译 CPU BF16 算子
# 第4个参数: GPU 架构（V100 = 70）

time MAX_JOBS=8 bash build.sh 1 python false "[70]" 2>&1 | tee "build_v100_$(date +%Y%m%d_%H%M%S).log"
```

编译完成后，产物位于：`FastDeploy/dist/`

**预期耗时**：约 20-40 分钟（仅供参考）

> **说明**：V100 编译时间比 A100/H100 短很多，因为跳过了大量 SM80+/SM89+/SM90+ 专用算子的编译（如 append_attention、FP8 GEMM、MLA 等）。实际耗时取决于机器配置。

### Step 2：二次编译测试

初次编译时间较长，二次编译因为有编译缓存的存在，时间会缩短。让我们来感受下修改不同文件的二次编译时间：

- 修改 kernel_traits 头文件：`custom_ops/gpu_ops/flash_mask_attn/kernel_traits.h`
- 修改 transfer_output 的 cc 文件：`custom_ops/gpu_ops/transfer_output.cc`
- 修改 python 文件：`custom_ops/gpu_ops/read_ids.py`

二次编译方式：对应文件加一个空行/空格保存退出后，执行：

```bash
time MAX_JOBS=8 bash build.sh 0 python false "[70]" 2>&1 | tee "rebuild_v100_$(date +%Y%m%d_%H%M%S).log"
```

### Step 3：安装 whl 包

```bash
pip install dist/fastdeploy*.whl
```

### Step 4：验证 V100 支持

```bash
python -c "
from fastdeploy.platforms import current_platform
print(f'Platform: {current_platform}')
sm_version = current_platform.get_sm_version()
print(f'SM Version: {sm_version}')
print(f'Is V100 (SM70): {sm_version == 70}')"
```

**预期输出**：

```
Platform: <fastdeploy.platforms.cuda.CUDAPlatform object at 0x...>
SM Version: 70
Is V100 (SM70): True
```

### Step 5：运行单元测试

根据 PR #6306 的修改，V100 (SM70) 当前支持以下功能：
- **cc >= 70**: W8A8 量化、MOE GEMM（非 FP8）、speculate_decoding、基础算子
- **跳过**: append_attention、MLA、FP8 相关（需要 SM80+/SM89+）

> **提示**：更多算子的 V100 兼容性支持正在开发中，请关注 PR #6306 获取最新进展。

PR #6306 在 Python 层添加了运行时兼容性处理：
- 算子导入使用 try-except，缺失时不会崩溃
- 测试文件添加了 `@unittest.skipIf` 装饰器，SM 版本不足时自动跳过

#### 安装测试依赖

```bash
# 如果没有安装 pytest，需要先安装
pip install pytest
```

#### 可运行的测试（推荐）

> **提示**：如果没有 pytest，可以用 `python <test_file.py>` 直接运行（测试文件基于 unittest 框架）。

```bash
# 设置日志文件
TEST_LOG="test_v100_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a $TEST_LOG
echo "[$(date)] V100 单元测试开始" | tee -a $TEST_LOG
echo "========================================" | tee -a $TEST_LOG

# 1. Platform 测试（V100 会有 1 个 fallback 相关的测试失败，属于预期行为）
echo "" | tee -a $TEST_LOG
echo "[$(date)] >>> 1. Platform 测试" | tee -a $TEST_LOG
python tests/platforms/test_platforms.py 2>&1 | tee -a $TEST_LOG
echo "[$(date)] <<< Platform 测试完成" | tee -a $TEST_LOG

# 2. 基础算子测试（cc >= 70 基础源文件）
echo "" | tee -a $TEST_LOG
echo "[$(date)] >>> 2. 基础算子测试" | tee -a $TEST_LOG
python tests/operators/test_rebuild_padding.py 2>&1 | tee -a $TEST_LOG
python tests/operators/test_get_padding_offset.py 2>&1 | tee -a $TEST_LOG
python tests/operators/test_set_value_by_flags_and_idx.py 2>&1 | tee -a $TEST_LOG
python tests/operators/test_stop_generation_multi_ends.py 2>&1 | tee -a $TEST_LOG
python tests/operators/test_token_penalty.py 2>&1 | tee -a $TEST_LOG
python tests/operators/test_fused_rotary_position_encoding.py 2>&1 | tee -a $TEST_LOG
python tests/operators/test_gelu_tanh.py 2>&1 | tee -a $TEST_LOG
echo "[$(date)] <<< 基础算子测试完成" | tee -a $TEST_LOG

# 3. Speculate Decoding 测试（cc >= 70）
echo "" | tee -a $TEST_LOG
echo "[$(date)] >>> 3. Speculate Decoding 测试" | tee -a $TEST_LOG
python tests/operators/test_speculate_update.py 2>&1 | tee -a $TEST_LOG
python tests/operators/test_speculate_verify.py 2>&1 | tee -a $TEST_LOG
echo "[$(date)] <<< Speculate Decoding 测试完成" | tee -a $TEST_LOG

# 4. MOE 测试（cc >= 70，跳过 FP8）
echo "" | tee -a $TEST_LOG
echo "[$(date)] >>> 4. MOE 测试" | tee -a $TEST_LOG
python tests/operators/test_moe_top_k_select.py 2>&1 | tee -a $TEST_LOG
python tests/operators/test_tritonmoe_preprocess.py 2>&1 | tee -a $TEST_LOG
echo "[$(date)] <<< MOE 测试完成" | tee -a $TEST_LOG

# 5. Sampling 测试
echo "" | tee -a $TEST_LOG
echo "[$(date)] >>> 5. Sampling 测试" | tee -a $TEST_LOG
python tests/operators/test_top_k_renorm_probs.py 2>&1 | tee -a $TEST_LOG
python tests/operators/test_rejection_top_p_sampling.py 2>&1 | tee -a $TEST_LOG
echo "[$(date)] <<< Sampling 测试完成" | tee -a $TEST_LOG

# 6. FFN 层测试（已内置 V100 兼容，自动检测 SM 版本）
# 预期输出: current sm_version=70, Disabling quantization for V100
echo "" | tee -a $TEST_LOG
echo "[$(date)] >>> 6. FFN 层测试" | tee -a $TEST_LOG
python tests/layers/test_ffn.py 2>&1 | tee -a $TEST_LOG
echo "[$(date)] <<< FFN 层测试完成" | tee -a $TEST_LOG

# 7. 以下测试会自动跳过（已添加 @unittest.skipIf 装饰器）
echo "" | tee -a $TEST_LOG
echo "[$(date)] >>> 7. 自动跳过测试 (SM89+)" | tee -a $TEST_LOG
python tests/layers/test_attention_layer.py 2>&1 | tee -a $TEST_LOG   # 会显示 skipped
python tests/layers/test_fusedmoe.py 2>&1 | tee -a $TEST_LOG          # 会显示 skipped
python tests/quantization/test_w4afp8.py 2>&1 | tee -a $TEST_LOG      # 会显示 skipped
echo "[$(date)] <<< 自动跳过测试完成" | tee -a $TEST_LOG

echo "" | tee -a $TEST_LOG
echo "========================================" | tee -a $TEST_LOG
echo "[$(date)] V100 单元测试全部完成" | tee -a $TEST_LOG
echo "测试日志已保存到: $TEST_LOG" | tee -a $TEST_LOG
echo "========================================" | tee -a $TEST_LOG

# 统计测试结果
echo "" | tee -a $TEST_LOG
echo ">>> 测试结果统计 <<<" | tee -a $TEST_LOG
grep -E "^(OK|FAILED|Ran)" $TEST_LOG | tee -a $TEST_LOG

```

#### V100 测试结果总结

基于实际 V100 测试，以下是各测试的通过情况：

| 测试类别 | 测试文件 | 通过 | 跳过 | 失败 | 说明 |
|---------|---------|:----:|:----:|:----:|------|
| **Platform** | `test_platforms.py` | 28 | 0 | 1 | fallback 测试失败（预期行为） |
| **基础算子** | `test_rebuild_padding.py` | 2 | 0 | 0 | |
| | `test_get_padding_offset.py` | 1 | 0 | 0 | |
| | `test_set_value_by_flags_and_idx.py` | 6 | 0 | 0 | |
| | `test_stop_generation_multi_ends.py` | 2 | 0 | 0 | |
| | `test_token_penalty.py` | 4 | 0 | 0 | |
| | `test_fused_rotary_position_encoding.py` | 1 | 0 | 0 | |
| | `test_gelu_tanh.py` | 1 | 0 | 0 | |
| **Speculate Decoding** | `test_speculate_update.py` | 2 | 0 | 0 | |
| | `test_speculate_verify.py` | 4 | 0 | 0 | |
| **MOE** | `test_moe_top_k_select.py` | 2 | 0 | 0 | |
| **FFN** | `test_ffn.py` | 1 | 0 | 0 | 自动禁用 FP8 量化 |
| **自动跳过** | `test_attention_layer.py` | 0 | 4 | 0 | SM89+ |
| | `test_fusedmoe.py` | 0 | 1 | 0 | SM89+ |
| | `test_w4afp8.py` | 6 | 5 | 0 | FP8 测试跳过 |

**总计**：通过 60+，跳过 10，失败 1（预期）

#### 需要跳过的测试

以下测试依赖 SM80+/SM89+/SM90+ 算子：

| 类别 | 测试文件 | V100 行为 |
|------|---------|----------|
| **自动跳过 (有 skipIf)** | `test_attention_layer.py` | 显示 skipped (SM89+) |
| | `test_fusedmoe.py` | 显示 skipped (SM89+) |
| | `test_w4afp8.py` | 显示 skipped (SM89+) |
| **CUTLASS/INT8 (SM75+/SM80+)** | `test_dequant.py` | ldmatrix 指令不支持 V100 |
| | `test_cutlass_scaled_mm.py` | BF16 GEMM 需要 SM80+ |
| **需手动跳过** | `test_append_attention.py` | NotImplementedError |
| | `test_plas_attention.py` | ImportError |
| | `test_flash_mask_attn.py` | NotImplementedError (SM90+) |
| | `test_moba_attention_backend.py` | ImportError (SM90+) |
| **FP8 相关 (SM89+)** | `test_per_token_quant.py` | ImportError |
| | `test_fp8_*.py` | ImportError |
| | `test_dynamic_per_token_scaled_fp8_quant.py` | ImportError |
| | `test_fused_hadamard_quant_fp8.py` | ImportError |
| **Hopper (SM90+)** | `test_machete_mm.py` | ImportError |
| | `test_w4afp8_gemm.py` | ImportError |

---

## 📧 邮件格式

**标题**：[Hackathon-FastDeploy V100 热身打卡]

**内容**：

```
飞桨团队你好，

【GitHub ID】：XXX

【打卡内容】：V100 初次编译/二次编译/安装whl包/运行单元测试

【打卡截图】：
```

| 项目 | 内容 |
|------|------|
| 硬件 | V100 (SM70), CUDA 11.x, 32GB 内存<br/><img width="764" height="290" alt="Image" src="https://github.com/user-attachments/assets/87e7f60c-89ec-45de-9607-1ab37e286536" /> |
| 编译分支 | PR #6306, commit: `48adbc40fc29d0cd660311d141eff0ca48f037d2`<br/>编译方式参考[源码编译文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/nvidia_gpu.md#4-wheel包源码编译) |
| 初次编译命令和时间 | 命令：`time MAX_JOBS=8 bash build.sh 1 python false [70]`<br/>时间：以下时间仅作为示例，不代表真实的初次编译时间（V100 跳过大量算子，时间较短仅供参考）<br/><img width="816" height="732" alt="Image" src="https://github.com/user-attachments/assets/be38522e-cd13-4cf3-afe2-295ecdc4cab6" /> |
| 二次编译时间 | 时间：以下时间仅作为示例，不代表真实的初次编译时间<br/>`custom_ops/gpu_ops/flash_mask_attn/kernel_traits.h`<br/>`custom_ops/gpu_ops/transfer_output.cc`<br/>`custom_ops/gpu_ops/read_ids.py`<br/><img width="805" height="169" alt="Image" src="https://github.com/user-attachments/assets/e20e32e1-59d4-48f6-af22-62efa96575c0" /> |
| 安装whl包 | 截图<br/><img width="813" height="384" alt="Image" src="https://github.com/user-attachments/assets/b645bf0f-4102-475f-b517-d000664416a0" /> |
| SM Version 验证 | SM Version: 70, Is V100: True |
| 运行单元测试 | <img width="814" height="555" alt="Image" src="https://github.com/user-attachments/assets/1a13b76a-c2ef-4b7e-9ef6-34fa42da38d0" /> |

---

## ❓ V100 常见问题

### 1. 编译被 Killed (OOM)

**原因**：nvcc 并发编译消耗大量内存

**解决**：

```bash
# 降低并发数
MAX_JOBS=4 bash build.sh 1 python false "[70]"

# 或更保守
MAX_JOBS=2 bash build.sh 1 python false "[70]"
```

### 2. 残留进程清理

```bash
pkill -9 nvcc; pkill -9 cc1plus; pkill -9 cicc; pkill -9 ptxas
rm -rf custom_ops/build custom_ops/tmp build *.egg-info dist
```

### 3. cuda::std::numeric_limits 编译错误

**错误信息**：
```
gpu_ops/sample_kernels/sampling.cuh(748): error: name followed by "::" must be a class or namespace name
```

**原因**：`cuda::std::numeric_limits` 是 libcu++ 特性，需要 SM80+ 架构支持

**解决**：此问题已在 PR #6306 中修复，请确保使用锁定的 commit

### 4. test_ffn.py 运行报错

`test_ffn.py` 已内置 V100 兼容性处理，会自动检测 SM 版本：
- SM >= 80: 使用 bfloat16 + BlockWiseFP8Config
- SM < 80 (V100): 使用 float16 + 禁用 FP8 量化

如果仍然报错，可能是其他原因，请检查：
1. PaddlePaddle 版本是否正确（需要 3.x）
2. 是否正确安装了 FastDeploy whl 包
3. 查看具体错误信息

### 5. APPEND_ATTN / MLA 相关测试失败

PR #6306 在 Python 层添加了运行时兼容性处理：

**情况 1：NotImplementedError**
```
NotImplementedError: append_attention is not available on this GPU architecture (requires SM80+).
V100 (SM70) does not support this operation.
```
这是预期行为，说明算子在 V100 上不可用。

**情况 2：ImportError**
```
ImportError: cannot import name 'xxx' from 'fastdeploy.model_executor.ops.gpu'
```
算子在编译时被跳过，导入失败。

**解决**：跳过这些测试，它们在 V100 上不适用。部分测试已添加 `@unittest.skipIf` 装饰器会自动跳过。

### 6. 链接错误：No such file or directory

**错误信息**：

```
x86_64-linux-gnu-g++: error: .../xxx.cu.o: No such file or directory
```

**原因**：之前编译被中断或部分文件编译失败

**解决**：完全清理构建缓存后重新编译

```bash
rm -rf custom_ops/build custom_ops/tmp build *.egg-info dist
MAX_JOBS=8 bash build.sh 1 python false "[70]" 2>&1 | tee "build_v100_$(date +%Y%m%d_%H%M%S).log"
```

---

## 📝 完整一键命令

从零开始的完整流程，可直接复制执行：

```bash
#!/bin/bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BUILD_LOG="build_v100_${TIMESTAMP}.log"
TEST_LOG="test_v100_${TIMESTAMP}.log"

echo "[$(date)] === 开始 V100 编译流程 ===" | tee $BUILD_LOG

# 1. 清理残留进程
pkill -9 nvcc 2>/dev/null || true
pkill -9 cc1plus 2>/dev/null || true

# 2. 安装 PaddlePaddle (V100 使用 CUDA 11.8)
echo "[$(date)] === 安装 PaddlePaddle ===" | tee -a $BUILD_LOG
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ 2>&1 | tee -a $BUILD_LOG

# 3. 克隆代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# 4. 切换到 PR #6306 并锁定 commit
git fetch origin pull/6306/head:pr-6306
git checkout pr-6306
git reset --hard 48adbc40fc29d0cd660311d141eff0ca48f037d2
echo "[$(date)] Commit: $(git log --oneline -1)" | tee -a ../$BUILD_LOG

# 5. 编译 (V100 = SM70)
echo "[$(date)] === 开始编译 ===" | tee -a ../$BUILD_LOG
time MAX_JOBS=8 bash build.sh 1 python false "[70]" 2>&1 | tee -a ../$BUILD_LOG

# 6. 安装 FastDeploy
echo "[$(date)] === 安装 FastDeploy ===" | tee -a ../$BUILD_LOG
pip install dist/fastdeploy*.whl 2>&1 | tee -a ../$BUILD_LOG

# 7. 验证 SM 版本
echo "[$(date)] === 验证 SM 版本 ===" | tee -a ../$BUILD_LOG
python -c "
from fastdeploy.platforms import current_platform
sm_version = current_platform.get_sm_version()
print(f'Platform: {current_platform}')
print(f'SM Version: {sm_version}')
print(f'Is V100 (SM70): {sm_version == 70}')
" 2>&1 | tee -a ../$BUILD_LOG

# 8. 运行单元测试
echo "[$(date)] === 运行单元测试 ===" | tee -a ../$TEST_LOG
echo "========================================" | tee -a ../$TEST_LOG

# Platform 测试
echo "" | tee -a ../$TEST_LOG
echo "[$(date)] >>> Platform 测试" | tee -a ../$TEST_LOG
python tests/platforms/test_platforms.py 2>&1 | tee -a ../$TEST_LOG
echo "[$(date)] <<< Platform 测试完成" | tee -a ../$TEST_LOG

# 基础算子测试
echo "" | tee -a ../$TEST_LOG
echo "[$(date)] >>> 基础算子测试" | tee -a ../$TEST_LOG
python tests/operators/test_rebuild_padding.py 2>&1 | tee -a ../$TEST_LOG
python tests/operators/test_get_padding_offset.py 2>&1 | tee -a ../$TEST_LOG
python tests/operators/test_set_value_by_flags_and_idx.py 2>&1 | tee -a ../$TEST_LOG
python tests/operators/test_stop_generation_multi_ends.py 2>&1 | tee -a ../$TEST_LOG
python tests/operators/test_token_penalty.py 2>&1 | tee -a ../$TEST_LOG
python tests/operators/test_fused_rotary_position_encoding.py 2>&1 | tee -a ../$TEST_LOG
python tests/operators/test_gelu_tanh.py 2>&1 | tee -a ../$TEST_LOG
echo "[$(date)] <<< 基础算子测试完成" | tee -a ../$TEST_LOG

# Speculate Decoding 测试
echo "" | tee -a ../$TEST_LOG
echo "[$(date)] >>> Speculate Decoding 测试" | tee -a ../$TEST_LOG
python tests/operators/test_speculate_update.py 2>&1 | tee -a ../$TEST_LOG
python tests/operators/test_speculate_verify.py 2>&1 | tee -a ../$TEST_LOG
echo "[$(date)] <<< Speculate Decoding 测试完成" | tee -a ../$TEST_LOG

# MOE 测试
echo "" | tee -a ../$TEST_LOG
echo "[$(date)] >>> MOE 测试" | tee -a ../$TEST_LOG
python tests/operators/test_moe_top_k_select.py 2>&1 | tee -a ../$TEST_LOG
python tests/operators/test_tritonmoe_preprocess.py 2>&1 | tee -a ../$TEST_LOG
echo "[$(date)] <<< MOE 测试完成" | tee -a ../$TEST_LOG

# Sampling 测试
echo "" | tee -a ../$TEST_LOG
echo "[$(date)] >>> Sampling 测试" | tee -a ../$TEST_LOG
python tests/operators/test_top_k_renorm_probs.py 2>&1 | tee -a ../$TEST_LOG
python tests/operators/test_rejection_top_p_sampling.py 2>&1 | tee -a ../$TEST_LOG
echo "[$(date)] <<< Sampling 测试完成" | tee -a ../$TEST_LOG

# FFN 层测试
echo "" | tee -a ../$TEST_LOG
echo "[$(date)] >>> FFN 层测试" | tee -a ../$TEST_LOG
python tests/layers/test_ffn.py 2>&1 | tee -a ../$TEST_LOG
echo "[$(date)] <<< FFN 层测试完成" | tee -a ../$TEST_LOG

# 自动跳过的测试 (SM89+)
echo "" | tee -a ../$TEST_LOG
echo "[$(date)] >>> 自动跳过的测试 (SM89+)" | tee -a ../$TEST_LOG
python tests/layers/test_attention_layer.py 2>&1 | tee -a ../$TEST_LOG
python tests/layers/test_fusedmoe.py 2>&1 | tee -a ../$TEST_LOG
python tests/quantization/test_w4afp8.py 2>&1 | tee -a ../$TEST_LOG
echo "[$(date)] <<< 自动跳过的测试完成" | tee -a ../$TEST_LOG

# 测试结果统计
echo "" | tee -a ../$TEST_LOG
echo "========================================" | tee -a ../$TEST_LOG
echo ">>> 测试结果统计 <<<" | tee -a ../$TEST_LOG
grep -E "^(OK|FAILED|Ran)" ../$TEST_LOG | tee -a ../$TEST_LOG
echo "========================================" | tee -a ../$TEST_LOG

echo "[$(date)] === 全部完成 ===" | tee -a ../$BUILD_LOG
echo "编译日志: $BUILD_LOG"
echo "测试日志: $TEST_LOG"
```

---

## 🔗 参考链接

- [PR #6306: V100 支持](https://github.com/PaddlePaddle/FastDeploy/pull/6306) - 欢迎关注进展、提交反馈
- [A100 热身打卡教程](https://github.com/PaddlePaddle/FastDeploy/issues/6225)
- [FastDeploy 源码编译文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/zh/get_started/installation/nvidia_gpu.md)

---

## 🙏 致谢

感谢参与 V100 兼容性开发和测试的所有贡献者！

V100 支持功能正在持续完善中，如果你在测试过程中遇到问题或有改进建议，欢迎在 [PR #6306](https://github.com/PaddlePaddle/FastDeploy/pull/6306) 中反馈，也欢迎直接参与开发贡献！
