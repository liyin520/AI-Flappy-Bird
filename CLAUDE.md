# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Flappy Bird 🐦 — 用 NEAT 遗传算法训练 AI 自动玩 Flappy Bird。

项目路径：`D:\project1`

## Project Structure

```
├── CLAUDE.md        # Claude Code 项目说明
├── 项目计划书.md     # 项目计划
├── main.py          # 入口（根目录）
├── src/             # 源代码
│   ├── game.py     # 游戏核心逻辑
│   └── config.py   # 配置参数
├── models/          # 训练好的 AI 模型
├── requirements.txt # 依赖清单
├── README.md        # 项目说明
└── .git/            # Git 仓库
```

## Commands

### 运行（conda 环境 flappybird）
- `python main.py` — 启动游戏（手动模式）
- `python main.py --train` — 启动 AI 训练模式
- `python main.py --replay` — 回放最佳 AI

### 环境
- 激活环境：`conda activate flappybird`
- 创建环境：`conda create -n flappybird python=3.10`
- 安装依赖：`pip install -r requirements.txt`
- Python 路径：`C:\Users\nanfe\anaconda3\envs\flappybird\python.exe`

### Git
- `git add . && git commit -m "msg"` — 提交变更
