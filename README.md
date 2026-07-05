# AI Flappy Bird 🐦

用 NEAT 遗传算法训练 AI 自动玩 Flappy Bird。

## 效果展示

训练几代后，AI 就能从"啥也不会"进化到熟练穿过管道：

```
Gen 0  →  平均 fitness: 58     最好: 236
Gen 1  →  平均 fitness: 774    最好: 20900
...
```

> 每次训练结束后会显示 fitness 变化曲线图。

## 快速开始

### 环境要求
- Python 3.10+
- pygame 2.x
- neat-python 2.x

### 安装

```bash
# 推荐 conda 环境
conda create -n flappybird python=3.10 -y
conda activate flappybird

# 安装依赖
cd D:\project1
pip install -r requirements.txt
```

### 运行

```bash
# 手动模式 — 你自己玩（空格/鼠标跳跃）
python main.py

# AI 训练模式
python main.py --train

# 回放已训练好的最佳 AI
python main.py --replay
```

## 项目结构

```
├── main.py              # 入口
├── src/
│   ├── config.py        # 游戏参数
│   ├── game.py          # 游戏核心逻辑
│   ├── ai_trainer.py    # AI 训练 + 回放
│   └── neat_config.txt  # NEAT 配置
├── models/              # 训练好的模型
├── requirements.txt     # 依赖清单
└── README.md
```

## 技术原理

### NEAT 算法
[NEAT](https://neat-python.readthedocs.io/)（NeuroEvolution of Augmenting Topologies）是一种通过**遗传算法**进化神经网络的算法：

1. **种群** — 100 个神经网络同时开始
2. **评估** — 每个网络玩一次游戏，得分作为适应度
3. **选择** — 表现好的网络有更高概率繁殖后代
4. **交叉 + 变异** — 结合两个父网络的基因，随机突变权重
5. **迭代** — 重复以上过程，种群逐代进化

### 神经网络设计

- **输入（6 个）**：小鸟 Y 坐标、速度、到下一个管道的水平距离、管道间隙中心 Y、间隙大小、到间隙中心的垂直偏移
- **输出（1 个）**：是否跳跃（0/1）
- **激活函数**：tanh

### 适应度函数

```
fitness = 得分 × 100 + 存活帧数
```

同时奖励"穿过管道"和"存活更久"。

## License

MIT
