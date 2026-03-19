# DAPO 训练前检查清单

## 📋 快速检查项

### 1. ✅ 数据格式验证

```bash
# 检查数据文件是否存在且格式正确
python -c "
import pandas as pd
df = pd.read_parquet('path/to/unified_math_25k.parquet')
print(f'✓ 数据集大小: {len(df)}')
print(f'✓ 字段: {df.columns.tolist()}')
print('\n检查第一个样本...')
print(df.iloc[0]['prompt'])
print(f'Ground truth: {df.iloc[0][\"reward_model\"][\"ground_truth\"]}')
"
```

**必须验证**：
- [ ] `prompt` 字段是 Qwen-Math 模板格式（包含 system, user, assistant）
- [ ] `reward_model.ground_truth` 存在且不为空
- [ ] `data_source` 字段正确设置

### 2. ✅ Tokenizer 兼容性

**Qwen 模型** 使用特殊 tokens: `<|im_start|>` 和 `<|im_end|>`

```bash
# 验证 tokenizer 是否正确识别这些 tokens
python -c "
from verl.utils import hf_tokenizer
tokenizer = hf_tokenizer('Qwen/Qwen2.5-Math-7B')
print('Tokenizer special tokens:')
print(f'  im_start: {tokenizer.im_start_id}')
print(f'  im_end: {tokenizer.im_end_id}')
print(f'  bos_token: {tokenizer.bos_token}')
print(f'  eos_token: {tokenizer.eos_token}')
"
```

**注意**：确保模型支持这些 tokens，否则需要调整模板格式。

### 3. ✅ Ground Truth 格式

**关键问题**：ground_truth 是原始答案（如 `"34"`）还是 boxed 格式（如 `"\\boxed{34}"`）？

```bash
# 检查 ground_truth 格式
python -c "
import pandas as pd
df = pd.read_parquet('path/to/unified_math_25k.parquet')
for i in range(min(5, len(df))):
    gt = df.iloc[i]['reward_model']['ground_truth']
    print(f'Sample {i}: gt = \"{gt}\"')
"
```

**确保**：
- [ ] Ground truth 是**纯答案**（如 `"34"`），不是 `"\\boxed{34}"`
- [ ] 验证函数会从 response 的 `\boxed{}` 中提取答案来比较

### 4. ✅ 验证逻辑一致性

```bash
# 测试验证逻辑
python -c "
from verl.utils.reward_score.math_dapo import compute_score

# 测试用例
test_cases = [
    ('Answer: 34', '34'),  # 传统格式
    ('\\boxed{34}', '34'),  # Qwen-Math 格式
    ('The answer is \\boxed{34}', '34'),  # 混合格式
]

for response, gt in test_cases:
    result = compute_score(response, gt)
    print(f'Response: \"{response}\", GT: \"{gt}\" -> Score: {result[\"score\"]}')
"
```

**应该看到**：
- 只有包含 `\boxed{34}` 的 response 得分为 1.0（因为 `strict_box_verify=True`）

### 5. ✅ 批次大小配置

**避免之前的错误** (`num_gen_batches >= max_num_gen_batches`)

```bash
# 检查训练脚本中的批次大小设置
grep -n "train_prompt_bsz\|max_num_gen_batches\|n_resp_per_prompt" recipe/dapo/test_dapo_7b.sh
```

**关键配置**：
```bash
train_prompt_bsz=512          # 每批需要的有效prompt数
n_resp_per_prompt=16         # 每个prompt生成16个候选
max_num_gen_batches=100      # 最大重试次数（建议设置较大值）
```

**计算公式**：
- 每次生成：`batch_size × n_resp_per_prompt` 个候选响应
- 如果正确率 10%，每个 prompt 平均 1.6 个正确响应
- 需要约 `512 / 1.6 ≈ 320` 个 prompts 才能收集够
- 如果每批读取 32 个 prompts，需要约 10 次生成

### 6. ✅ 数据加载器配置

**检查**：训练时会读取多少 prompts？

```bash
# 检查 DAPO 训练脚本中的数据加载设置
python -c "
# 模拟数据加载
import pandas as pd
df = pd.read_parquet('path/to/unified_math_25k.parquet')
print(f'总样本数: {len(df)}')
print(f'预计批次大小: 512')
print(f'需要生成次数: {len(df) // 512}')
"
```

### 7. ✅ Reward 函数路径

**验证**：确保 reward 函数正确调用

```bash
# 检查 reward 函数配置
grep -n "reward_manager\|compute_score" recipe/dapo/test_dapo_7b.sh
```

应该看到：
```bash
reward_model.reward_manager=dapo
```

### 8. ✅ 特殊字符处理

**LaTeX 特殊字符**：
- `\boxed{}` → 确保 tokenizer 正确处理
- `\sqrt{}`, `\frac{}{}` 等 → 确保不会被 tokenizer 拆分错误

```bash
# 测试特殊字符编码
python -c "
from verl.utils import hf_tokenizer
tokenizer = hf_tokenizer('Qwen/Qwen2.5-Math-7B')

test_strings = [
    '\\boxed{34}',
    '\\sqrt{16}',
    '\\frac{1}{2}',
]

for s in test_strings:
    tokens = tokenizer.encode(s, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)
    print(f'{s} -> {len(tokens)} tokens -> {decoded}')
"
```

### 9. ✅ 内存和存储检查

```bash
# 检查数据集大小
du -h path/to/unified_math_25k.parquet

# 检查磁盘空间
df -h /path/to/save/checkpoints

# 检查模型大小
du -h /path/to/models/Qwen2.5-Math-7B
```

### 10. ✅ 环境变量

```bash
# 检查必要的环境变量
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "RAY_ADDRESS: $RAY_ADDRESS"
echo "NCCL settings: $NCCL_DEBUG"
```

## 🚨 常见问题

### 问题1：`num_gen_batches >= max_num_gen_batches`

**原因**：样本质量不够或批次大小设置不合理

**解决**：
```bash
# 降低批次大小
train_prompt_bsz=128  # 从 512 降到 128

# 或增加重试次数
max_num_gen_batches=0  # 无限制

# 或增加每批生成的响应数
n_resp_per_prompt=32  # 从 16 增加到 32
```

### 问题2：Ground truth 格式不匹配

**症状**：所有样本得分都是 -1.0

**解决**：
```bash
# 检查 ground_truth 是 "34" 还是 "\\boxed{34}"
# 确保是纯答案格式
```

### 问题3：Tokenizer 不支持特殊 tokens

**症状**：Tokenization 错误

**解决**：
```bash
# 使用支持 Qwen-Math 的模型
# 或修改模板格式为标准格式
```

## 📝 训练前最终检查

运行以下脚本进行一次完整验证：

```bash
#!/bin/bash
# final_check.sh

echo "=== 1. 数据检查 ==="
python -c "
import pandas as pd
df = pd.read_parquet('$DATASET_PATH')
print(f'✓ 数据大小: {len(df)}')
assert 'prompt' in df.columns
assert 'reward_model' in df.columns
"

echo "=== 2. Ground Truth 格式检查 ==="
python -c "
import pandas as pd
df = pd.read_parquet('$DATASET_PATH')
for i in range(min(3, len(df))):
    gt = df.iloc[i]['reward_model']['ground_truth']
    assert not gt.startswith('\\\\boxed'), f'GT {i} should not start with \\\\boxed'
    print(f'✓ Sample {i}: gt = {gt}')
"

echo "=== 3. Prompt 格式检查 ==="
python -c "
import pandas as pd
df = pd.read_parquet('$DATASET_PATH')
prompt = df.iloc[0]['prompt']
assert isinstance(prompt, list), 'Prompt should be a list'
assert len(prompt) == 3, 'Prompt should have 3 elements (system, user, assistant)'
print('✓ Prompt format correct')
"

echo "=== 4. Score 函数测试 ==="
python -c "
from verl.utils.reward_score.math_dapo import compute_score
result = compute_score('\\boxed{34}', '34')
assert result['score'] == 1.0, 'Score function not working'
print('✓ Score function correct')
"

echo "=== 所有检查通过！可以开始训练 ==="
```

## 🎯 训练启动命令

```bash
# 启动训练
python3 -m recipe.dapo.main_dapo \
    data.train_files="path/to/unified_math_25k.parquet" \
    data.val_files="path/to/test.parquet" \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    reward_model.reward_manager=dapo \
    actor_rollout_ref.rollout.n=16 \
    # ... 其他配置
```

## 📊 监控指标

训练时关注：
- `train/num_gen_batches` - 每个 step 生成的批次数
- `train/acc` - 正确率
- `train/score` - 平均奖励分数
- 如果 `num_gen_batches` 一直增加 → 样本质量不够
- 如果所有 score 都是 -1.0 → ground truth 格式问题

