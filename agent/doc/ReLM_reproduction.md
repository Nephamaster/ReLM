# ReLM（AAAI 2024）完整复现指南

> 论文：*Chinese Spelling Correction as Rephrasing Language Model*  
> Backbone：`bert-base-chinese`  
> 目标：复现 ECSpell 单任务、34M 合成数据零样本、LEMON/SIGHAN、Multi-task、FPR、线性探测和 Mask 消融。

---

## 1. 复现范围

论文的实验不只是一轮 ECSpell 微调，应拆成六条相互独立的实验线。

| 实验线 | 对应论文 | 初始化 | 训练数据 | 测试数据 |
|---|---|---|---|---|
| A. ECSpell 单任务 | Table 1 | `bert-base-chinese` | LAW / MED / ODW 各自训练集 | 同域测试集 |
| B. 34M ReLM 训练 | Table 2 的前置步骤 | `bert-base-chinese` | 3400 万单语或已生成纠错对 | 不在训练中选 LEMON |
| C. 零样本泛化 | Table 2 | 34M ReLM 权重 | 不再微调 | LEMON 七域 + SIGHAN |
| D. Multi-task | Table 3、4 | `bert-base-chinese` | ECSpell + TNEWS + AFQMC | 三个任务测试集 |
| E. 线性探测 | Table 5 | 先经过 CSC 训练的模型 | 冻结 encoder，仅训练 TNEWS 头 | TNEWS |
| F. Mask 消融 | Table 6、7 | `bert-base-chinese` | ECSpell | LAW / MED / ODW |

本轮就先完成 34M ReLM 训练

## 2. ReLM 核心目标

给定错误句子 \(X=(x_1,\ldots,x_n)\) 和正确句子 \(Y=(y_1,\ldots,y_n)\)，ReLM 构造：

```text
input : [CLS] x1 x2 ... xn [SEP] [MASK] [MASK] ... [MASK] [SEP]
label : [CLS] x1 x2 ... xn [SEP] y1     y2     ... yn     [SEP]
```

模型在后半段一次性填充完整正确句子，而不是在原句位置做字符到字符 tagging。

辅助 MFT 使用论文最优配置：

```text
mask_mode = noerror
mask_rate = 0.3
```

它只随机遮盖源句中的非错误位置。训练 loss 包含：

1. 后半段全部 `[MASK]` 位置的完整目标句生成损失；
2. 源句被辅助 Mask 遮盖位置的恢复损失；
3. 其他位置设为 `-100`，不计 loss。

固定长度是 ReLM 的前提：源句和目标句必须等长。插入、删除、分词级改写样本不能直接进入训练。

## 3. 论文目标值

### 3.1 ECSpell，Table 1

| Domain | Precision | Recall | F1 |
|---|---:|---:|---:|
| LAW | 89.9 | 94.5 | 91.2 |
| MED | 79.2 | 85.9 | 82.4 |
| ODW | 82.4 | 84.8 | 83.6 |

仓库 README 另给出 `95.6 / 89.9 / 92.3`，与论文 Table 1 不一致。复现实验必须分别记录“论文目标”和“仓库版本结果”，不能混为同一配置。

### 3.2 LEMON + SIGHAN，Table 2

| GAM | ENC | COT | MEC | CAR | NOV | NEW | SIG | AVG |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 33.0 | 49.2 | 66.8 | 54.0 | 53.1 | 37.8 | 58.5 | 57.0 | 51.2 |

### 3.3 Mask Rate，Table 7

| Mask rate | LAW | MED | ODW | AVG |
|---:|---:|---:|---:|---:|
| 0% | 57.6 | 56.9 | 59.0 | 57.8 |
| 10% | 90.0 | 84.2 | 82.5 | 85.6 |
| 20% | 91.3 | 84.8 | 86.9 | 87.7 |
| 30% | 92.2 | 85.4 | 86.7 | 88.1 |
| 40% | 91.3 | 82.8 | 84.9 | 86.3 |
| 60% | 86.7 | 81.7 | 78.8 | 82.4 |

这些值用于判断复现是否处于合理区间，不应作为从测试集选择 checkpoint 的依据。

## 4. 推荐目录

```text
ReLM/
├── autocsc.py
├── run.py
├── run_relm.py
├── run_multi.py
├── run_relm_multi.py
├── confus/
├── data/
│   ├── tnews/
│   ├── rsighan/
│   ├── afqmc/
│   ├── ecspell/
│   │   ├── train_law.txt
│   │   ├── test_law.txt
│   │   ├── train_med.txt
│   │   ├── test_med.txt
│   │   ├── train_odw.txt
│   │   └── test_odw.txt
│   ├── lemon_v2/
│   │   ├── gam.txt
│   │   ├── enc.txt
│   │   ├── cot.txt
│   │   ├── mec.txt
│   │   ├── car.txt
│   │   ├── nov.txt
│   │   ├── new.txt
│   │   └── sig.txt
│   └── 34m_confuse_gen/
│       └──34m_confuse_gen.jsonl
├── tools/
│   ├── audit_relm_data.py
│   ├── train_relm_streaming.py
│   └── eval_relm.py
└── outputs/
```

## 5. 环境

核心依赖：

```text
torch
transformers
accelerate
numpy
tqdm
scikit-learn
scipy
jsonlines
```

注意：实验会在服务器上运行，本地只需撰写代码和 requirements.txt

### 5.1 已确认的服务器协议

本轮 34M 正式实验采用以下固定事实：

```text
完整数据：/share/project/wuhaiming/spaces/ReLM/data/34m_confuse_gen/34m_confuse_gen.jsonl
模型目录：/share/project/wuhaiming/data/models/bert-base-chinese
数据格式：每行一个 JSON 对象，字段为 src 和 tgt
训练输入：已生成纠错对，不再在线注错
训练配置：global batch=4096，max_train_steps=60000
mask_mode=noerror，mask_rate=0.3
continuous prompt：关闭
max_seq_length=128
```

本地 `data/34m_confuse_gen/34m_confuse_gen.example.jsonl` 仅用于检查字段格式，禁止作为正式训练输入。正式训练脚本默认拒绝文件名中包含 `.example.` 的文件。

训练启动时会把 git commit、模型与数据 SHA256、Tokenizer、Torch/Transformers/Accelerate 版本、GPU 名称、`CUDA_VISIBLE_DEVICES`、world size 和 global batch 写入 `train_config.json`，以固定服务器实验事实。

## 6. 数据说明

### 6.1 3400万数据

单语数据：

```text
发动机故障切忌盲目拆检
```

已生成纠错对：

```json
{"src": "发动机故障切纪盲目拆检", "tgt": "发动机故障切忌盲目拆检"}
```

原仓库 `ConfusDataset` 面向“每行一个干净句子”的单语格式，并在线注入一个错误。

但当前 data 目录中 34m_confuse_gen 存放的是 34m_confuse_gen.jsonl，格式详见 `data/34m_confuse_gen/34m_confuse_gen.example.jsonl`

因此在使用该数据集进行训练时，相应的代码需要做适配。

当前实现使用 `tools/train_relm_streaming.py` 的 `--data_mode pair` 读取上述 JSONL。它不会把 34M 数据加载到内存，也不会把本地 example 文件静默当作正式数据。单语在线注错模式仍作为兼容选项保留，但不属于本轮正式实验。

### 6.2 原始 ECSpell 数据已知问题

| 数据集 | 测试样本 | 长度超过约 62 字 |
|---|---:|---:|
| LAW | 500 | 24（4.8%） |
| MED | 500 | 141（28.2%） |
| ODW | 500 | 87（17.4%） |

`train_odw.txt` 第 514 行源句 22 字、目标句 24 字，会被原 processor 静默丢弃。

`max_seq_length=128` 时，标准 ReLM 模板只能容纳约 62 个目标 token：

```text
1 + n + 1 + n + 1 <= 128
n <= 62
```

因此 MED、ODW 的截断影响不可忽略。论文和代码默认 128，严格对齐时保留 128；另行报告 `max_seq_length=256` 的完整句结果，不能把两种配置混写。

## 7. 阶段 A：ECSpell 单任务

### 7.1 仓库原命令修正

README 命令遗漏了正确的 `data_dir`。应显式使用：

```bash
MODEL=/path/to/bert-base-chinese
DATA=$PWD/data/ecspell
OUT=$PWD/outputs/ecspell

for DOMAIN in law med odw; do
  CUDA_VISIBLE_DEVICES=0 python run_relm.py \
    --do_train \
    --do_eval \
    --data_dir "$DATA" \
    --load_model_path "$MODEL" \
    --mft \
    --mask_mode noerror \
    --mask_rate 0.3 \
    --prompt_length 1 \
    --train_on "$DOMAIN" \
    --eval_on "$DOMAIN" \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --max_train_steps 5000 \
    --train_batch_size 128 \
    --eval_batch_size 64 \
    --fp16 \
    --output_dir "$OUT/$DOMAIN"
done
```

注意：上面没有 `--apply_prompt`，因此连续 prompt 实际未启用。`prompt_length=1` 只是原代码关闭 prompt 时的占位方案。

### 7.2 论文要求的超参数搜索

论文不是固定只跑一组参数，而是在以下集合中选择：

```text
batch size ∈ {32, 128}
learning rate ∈ {2e-5, 5e-5}
steps = 5000
```

每个域应运行四组：

```bash
for DOMAIN in law med odw; do
  for BS in 32 128; do
    for LR in 2e-5 5e-5; do
      # 调用上面的 run_relm.py
    done
  done
done
```

### 7.3 两种结果协议

**仓库对齐协议**

- `get_dev_examples()` 直接读取 `test_{domain}.txt`；
- 每 100 step 在测试集上评测；
- 按测试 F1 选择最佳 checkpoint。

这种做法容易更接近仓库结果，但存在测试集泄漏。

**规范实验协议**

- 从 `train_{domain}.txt` 固定划分 10% 作为 dev；
- 四组超参数只根据 dev 选；
- 选定后重新训练或使用最佳 dev checkpoint；
- test 只运行一次。

报告中应同时给出：

```text
Repo-aligned result
Clean dev/test result
```

不得只报告测试集反复选优后的最高点。

### 7.4 BERTTagging 对照

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --do_train \
  --do_eval \
  --data_dir data/ecspell \
  --train_on train_law.txt \
  --eval_on test_law.txt \
  --load_model_path "$MODEL" \
  --model_type finetune \
  --learning_rate 5e-5 \
  --max_train_steps 5000 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --output_dir outputs/bert_tag_law \
  --fp16
```

MED、ODW 替换文件名。MFT tagging 增加：

```bash
--mft --noise_probability 0.3
```

## 8. 阶段 B：3400 万数据 ReLM 训练

### 8.1 为什么不建议直接用原 `run.py`

原代码存在四个阻断性问题：

1. `ConfusDataset` 虽然定义了，但训练主流程从未使用它；
2. `DataProcessor` 会把所有样本和全部 features 一次性加载到内存，3400 万规模不可行；
3. checkpoint 只在 `--do_eval` 分支内保存；不做中途评测时，长训练没有周期 checkpoint；
4. 只保存模型权重，不保存 optimizer、scheduler、step 和 RNG，无法严格断点续训。

应重新提供 `train_relm_streaming.py`，支持：

- 单语在线注错和已生成 pair 两种格式；
- 按 shard、rank、worker 流式读取；
- 标准全局 batch 计算；
- Accelerate DDP；
- 完整训练状态保存与恢复；
- 源句限定的辅助 Mask；
- Hugging Face 权重导出。

### 8.2 论文配置

论文正文：

```text
global batch = 4096
learning rate = 5e-5
steps = 60000
8 × A800
```

README：

```text
global batch = 8192
steps = 30000
```

两者处理的样本展示数相同：

```text
4096 × 60000 = 8192 × 30000 = 245,760,000
```

但优化步数不同，不是完全等价实验。严格复现论文时使用 4096 × 60000。

### 8.3 8 卡参考命令

假设每卡 micro batch 128，梯度累积 4：

```text
128 × 8 × 4 = 4096
```

```bash
accelerate launch --num_processes 8 tools/train_relm_streaming.py \
  --model_path /share/project/wuhaiming/data/models/bert-base-chinese \
  --data_glob /share/project/wuhaiming/spaces/ReLM/data/34m_confuse_gen/34m_confuse_gen.jsonl \
  --data_mode pair \
  --output_dir outputs/relm-34m-paper \
  --max_seq_length 128 \
  --per_device_train_batch_size 128 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 60000 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.06 \
  --mask_mode noerror \
  --mask_rate 0.3 \
  --mixed_precision fp16 \
  --save_steps 1000 \
  --keep_last_checkpoints 3 \
  --seed 42
```

该入口默认关闭 continuous prompt；本轮没有论文明确要求的 prompt 参数，因此不额外添加 prompt。正式协议要求 `world_size=8`，并按 `per_device_batch × world_size × gradient_accumulation=4096` 校验；smoke test 必须显式使用覆盖开关。

正式训练前先审计数据：

```bash
python tools/audit_relm_data.py \
  --data_glob /share/project/wuhaiming/spaces/ReLM/data/34m_confuse_gen/34m_confuse_gen.jsonl \
  --model_path /share/project/wuhaiming/data/models/bert-base-chinese \
  --output_file outputs/relm-34m-paper/data_audit.json
```

训练完成后评测 LEMON/SIGHAN，不能在这些测试集上继续微调：

```bash
python tools/eval_relm.py \
  --model_path outputs/relm-34m-paper/checkpoint-60000/hf_model \
  --test_dir data/lemon_v2 \
  --rsighan_dir data/rsighan \
  --categories gam,enc,cot,mec,car,nov,new,sig \
  --max_seq_length 128 \
  --batch_size 256 \
  --output_file outputs/relm-34m-paper/lemon_results.json
```

仓库中的 LEMON 七个域共 22,252 条；SIGHAN 数据位于 `data/rsighan/rSIGHAN13.jsonl`、`rSIGHAN14.jsonl` 和 `rSIGHAN15.jsonl`，评测脚本会将其合并为 `SIG`。由于 ReLM 要求源句和目标句等长，评测结果会在 JSON 中记录被过滤的非等长样本数量，不能将过滤后的样本数误认为原始数据总数。

### 8.4 断点续训

```bash
accelerate launch --num_processes 8 tools/train_relm_streaming.py \
  ...前述相同参数... \
  --resume_from outputs/relm-34m-paper/checkpoint-20000/accelerate_state
```

恢复后必须保持以下参数不变：

```text
world size
per-device batch
gradient accumulation
scheduler
数据 mode
tokenizer
max_seq_length
mask policy
```

### 8.5 生成策略对齐

原 `ConfusDataset.confus()` 每句选择一个汉字位置，替换来源比例为：

```text
同音集合        40%
近音集合        30%
形近集合        20%
随机字集合      10%
```

风险：

- 混淆集中包含重复项；
- 形近集合包含原字符自身；
- 某些生成样本实际上没有发生变化；
- 罕见字可能变成 `[UNK]`；
- 每句固定一个错误，与真实错误数分布不同。

为了论文对齐，第一轮保留原策略。后续可以清除 self-mapping 和重复项，但必须标记为改进实验。

## 9. 阶段 C：LEMON 与 SIGHAN 零样本

34M 模型训练完成后，不得在 LEMON 或 SIGHAN 上继续微调。

### 9.1 评测脚本

构建对应的评测脚本，以便能够：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval_relm.py \
  --model_path outputs/relm-34m-paper/checkpoint-60000/hf_model \
  --test_dir /path/to/lemon_v2 \
  --categories gam,enc,cot,mec,car,nov,new,sig \
  --max_seq_length 128 \
  --batch_size 256 \
  --output_file outputs/relm-34m-paper/lemon_results.json
```

### 9.2 使用原仓库评测

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --test_on_lemon /path/to/lemon_v2 \
  --output_dir outputs/lemon-original-eval \
  --model_type relm \
  --load_model_path /path/to/bert-base-chinese \
  --load_state_dict /path/to/relm-m0.3.bin \
  --max_seq_length 128 \
  --eval_batch_size 256 \
  --fp16
```

原 `run.py` 期望文件名：

```text
gam.txt car.txt nov.txt enc.txt new.txt cot.txt mec.txt sig.txt
```

平均值与迭代顺序无关，但最终表格应按论文顺序：

```text
GAM ENC COT MEC CAR NOV NEW SIG
```

## 10. 阶段 D：Multi-task（暂不做）

论文设置：

```text
任务：ECSpell + TNEWS + AFQMC
epoch：15
batch size：从 {32, 128} 选择
learning rate：从 {2e-5, 5e-5} 选择
```

LAW、MED、ODW 各训练一套 multi-task 模型。

### 10.1 原始命令

```bash
CUDA_VISIBLE_DEVICES=0 python run_relm_multi.py \
  --do_train \
  --do_eval \
  --data_dir data \
  --load_model_path /path/to/bert-base-chinese \
  --mft \
  --mask_mode noerror \
  --mask_rate 0.3 \
  --task_name "ecspell tnews afqmc" \
  --train_on "law base base" \
  --eval_on law \
  --csc_prompt_length 10 \
  --sent_prompt_length 3 \
  --save_steps 1000 \
  --learning_rate 5e-5 \
  --num_train_epochs 15 \
  --train_batch_size 128 \
  --eval_batch_size 64 \
  --fp16 \
  --output_dir outputs/multitask-law
```

MED、ODW 替换 `train_on` 第一项和 `eval_on`。

### 10.2 必须修正任务采样

论文写的是“从三个任务中均匀采样一个 batch”。原代码把所有任务样本拼接后使用 `RandomSampler`，实际采样概率近似与数据集大小成正比：

```text
ECSpell 约 2k
AFQMC 约 34k
TNEWS 约 53k
```

CSC 会被严重稀释，和论文描述不一致。

推荐实现为三个独立 DataLoader，每个 step 均匀选择任务：

```python
task_names = ["ecspell", "tnews", "afqmc"]
task_iters = {name: cycle(loader) for name, loader in loaders.items()}

for global_step in range(max_steps):
    task = task_names[global_step % len(task_names)]
    batch = next(task_iters[task])
    loss = model(**batch)[0]
    loss.backward()
```

使用固定轮转与均匀随机在长期频率上都满足 1:1:1；固定轮转更容易复现。

### 10.3 Multi-task 评测

原脚本一次只评估 `task_names[0]`。同一 checkpoint 需要分别评估：

```text
ECSpell
TNEWS
AFQMC
```

同时保留完整三个 verbalizer 和模型结构，加载 checkpoint 时先检查：

```text
missing_keys == []
unexpected_keys == []
```

使用 `strict=False` 但不检查 key 会导致“命令成功、权重实际未加载”的隐性错误。

---

## 11. 阶段 E：线性探测（先不做）

Table 5 的流程：

1. 在 LAW、MED、ODW 上分别训练 CSC 模型；
2. 冻结 CSC 训练后的 encoder；
3. 初始化新的 TNEWS 线性分类器；
4. 只训练分类器；
5. 与直接用原始 BERT 做 TNEWS 的结果比较。

ReLM 应使用和 `MultiTaskReLM` 兼容的 CSC checkpoint。不要直接把 `run_relm.py` 的 `PTuningWrapper` checkpoint 填给 `run_relm_multi.py`，两者 state dict 前缀和结构不同。

推荐先用：

```bash
python run_relm_multi.py \
  --do_train \
  --task_name ecspell \
  --train_on law \
  ...其余 CSC 参数...
```

再执行仓库 README 的 linear probing 命令：

```bash
CUDA_VISIBLE_DEVICES=0 python run_relm_multi.py \
  --do_train \
  --do_eval \
  --load_state_dict /path/to/compatible-csc-checkpoint.bin \
  --task_name tnews \
  --train_on base \
  --eval_on base \
  --csc_prompt_length 10 \
  --sent_prompt_length 3 \
  --save_steps 1000 \
  --learning_rate 5e-5 \
  --max_train_steps 5000 \
  --train_batch_size 128 \
  --eval_batch_size 64 \
  --fp16 \
  --output_dir outputs/linear-probe-law \
  --freeze_lm \
  --not_apply_prompt \
  --linear_prob
```

Table 5 的 ReLM 目标：

| 来源 | TNEWS F1 |
|---|---:|
| LAW → TNEWS | 54.1 |
| MED → TNEWS | 53.7 |
| ODW → TNEWS | 49.2 |
| 原始 BERT → TNEWS | 56.6 |

---

## 12. 阶段 F：Mask 消融（先不做）

论文 Table 6、7 说明这些结果基于带 learnable prompt 的 ReLM。仓库没有明确报告单任务 prompt 的唯一长度，存在不可消除的配置缺口。

### 12.1 Mask 策略

```bash
# mask any
--mft --mask_mode all --mask_rate 0.3 --apply_prompt

# mask non-error
--mft --mask_mode noerror --mask_rate 0.3 --apply_prompt
```

### 12.2 Mask Rate

```bash
for RATE in 0.0 0.1 0.2 0.3 0.4 0.6; do
  # LAW、MED、ODW 分别训练
done
```

`run.py` 对 `model_type=relm` 无论是否传 `--mft` 都会调用辅助 masking。因此 0% 消融必须显式设置：

```bash
--noise_probability 0.0
```

不能只删除 `--mft`。

### 12.3 Prompt 配置记录

至少记录：

```text
apply_prompt
prompt_length
是否训练全部 BERT 参数
是否从相同初始化开始
```

建议先跑 `prompt_length ∈ {3, 10}` 的小规模验证，并把最终选择写入实验配置。不要用测试集选择 prompt 长度。

---

## 13. Table 4 与 Figure 3

FPR 定义：

```text
FPR = 被错误修改的正确句数 / 正确句总数
```

同一 checkpoint 的评测输出已经包含 FPR。论文 ReLM 单任务目标：

| LAW | MED | ODW | AVG |
|---:|---:|---:|---:|
| 8.4 | 5.0 | 6.9 | 6.8 |

Figure 3 不需要重新训练不同正负比例模型。可根据固定 recall 和 FPR 改变 \(r=N/P\)：

```python
precision = recall / (recall + fpr * r)
f1 = 2 * precision * recall / (precision + recall)
```

绘图时明确 `recall`、`fpr` 使用小数而不是百分数。

---

## 14. 完整实验记录模板

每次运行保存：

```yaml
experiment_id:
git_commit: 3d1beb9
paper_or_repo_protocol:
model_path:
model_hash:
tokenizer_hash:
data_files:
data_hash:
train_samples:
dropped_length_mismatch:
dropped_token_mismatch:
unk_rate:
max_seq_length:
mask_mode:
mask_rate:
prompt_enabled:
prompt_length:
per_device_batch:
world_size:
gradient_accumulation:
global_batch:
learning_rate:
scheduler:
warmup_ratio:
max_steps_or_epochs:
seed:
checkpoint:
dev_metrics:
test_metrics:
```

建议 seed：

```text
42, 43, 44
```

主结果报告均值和标准差，同时附单次最佳值，避免只展示偶然高点。

## 15. 代码级风险清单

### P0：会阻断或严重歪曲实验

1. **34M 无法直接训练**：原 `run.py` 没有接入 `ConfusDataset`，并把数据全量加载内存。
2. **无可靠断点续训**：原脚本只保存 model state，不保存 optimizer、scheduler、step、RNG。
3. **测试集泄漏**：ECSpell 的 dev 和 test 指向同一文件。
4. **Multi-task 采样不均匀**：原代码按样本量采样，不是论文的任务均匀采样。
5. **长度截断严重**：`max_seq_length=128` 时最多约 62 个目标 token。
6. **论文与 README 配置冲突**：`4096×60000` 对 `8192×30000`；ECSpell 结果也不一致。
7. **checkpoint 架构不兼容**：`run_relm.py`、`run.py`、`run_relm_multi.py` 的 wrapper 和 key 前缀不同。

### P1：可能导致结果偏差

8. `run.py` 的 ReLM 分支忽略 `--mft` 开关，始终执行辅助 Mask。
9. `run_relm.py` 使用额外 `[CLS]/[SEP]` 作为 prompt 占位，同时 tokenizer 还会自动添加外围 special tokens。
10. `run.py` 中存在无说明的 `token_id == 6` 删除逻辑。
11. `skip_special_tokens=True` 会隐藏 `[UNK]` 等 token，可能改变评测序列长度。
12. 混淆集包含重复项、自映射和罕见字，生成数据可能无变化或产生 `[UNK]`。
13. 原脚本多卡参数含义不标准。`run.py` 先将 `train_batch_size` 除以梯度累积，再交给 Accelerate。
14. fp16、随机 Mask、DataLoader 顺序和 cudnn 会造成波动；仅设置 seed 不能保证完全确定。
15. 原指标实现某些分母没有统一做零保护，极端 checkpoint 可能报除零。

### P2：对照实验风险

16. `run_gpt.py` 的生成评测存在边界和列表追加问题，最后一个样本可能被漏掉，输入列表也会重复追加。
17. ChatGPT 10-shot 依赖历史在线模型，不存在固定可下载权重。
18. Baichuan2 LoRA 的训练环境、模板和 checkpoint 版本需单独冻结。
19. Case Study 是人工挑选案例，不应当作为独立定量结论。
20. LEMON 数据许可限制需要在实验数据管理中保留说明。

## 16. 推荐执行顺序

```text
1. 冻结代码、模型、Tokenizer 和环境版本
3. LAW 100-step smoke test
4. LAW/MED/ODW 四组超参数网格
5. BERTTagging 对照
6. 34M streaming 100-step 吞吐与断点测试
7. 34M 正式 60000-step 训练
8. LEMON + SIGHAN 零样本评测
9. Multi-task 任务均匀采样修正版
10. FPR、线性探测、Mask 策略与比例消融
11. 3 seeds 汇总、错误案例分析、复现偏差说明
```

### Smoke test 验收

```text
loss 有限且下降
每个 batch 的 source/output slot 数量相等
只在输出 slot 和辅助 Mask 位置产生 label
保存 checkpoint 后能恢复并继续
单卡与多卡的 global batch 计算正确
同一 checkpoint 重复评测结果一致
```

## 17. 最终产物

建议最终目录包含：

```text
outputs/
├── ecspell/
│   ├── law/
│   ├── med/
│   └── odw/
├── relm-34m-paper/
│   ├── train_config.json
│   ├── checkpoint-*/
│   └── lemon_results.json
├── multitask/
├── linear-probe/
├── ablation/
├── metrics_summary.csv
└── reproduction_report.md
```

主报告中必须明确区分：

```text
论文原始配置
仓库实际配置
为修复工程缺陷所做的改动
严格复现结果
规范无泄漏结果
```

这样即使最终数值无法完全对齐，也能判断偏差来自数据、代码版本、采样、模型选择还是训练随机性。
