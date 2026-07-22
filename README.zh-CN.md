<div align="center">
  <img src="docs/assets/banner.svg" alt="Cortex Arena 分层策略游戏智能体" width="100%">

  <h3>一个能探索、记忆、规划并直接进行策略对局的可复现游戏智能体</h3>

  [English](README.md) · [架构](docs/architecture.md) · [调研记录](docs/research.md)
</div>

<div align="center">
  <img src="docs/assets/gameplay.png" alt="Cortex Arena 真实策略对局画面" width="640">
  <br>
  <sub>种子 42 的真实回放第 253/505 帧，由 Lux S3 官方可视化器渲染；蓝方为 Cortex Arena，红方为基线。</sub>
</div>

## 项目定位

Cortex Arena 运行在开源的 [Lux AI Season 3](https://github.com/Lux-AI-Challenge/Lux-Design-S3) 环境中。这里不是一个“随机移动几步”的示例：游戏包含战争迷雾、同步行动、持续五局的地图记忆、随机机制、能量管理、隐藏得分格和范围战斗，并能生成可在浏览器观看的完整回放。

我选择这条路线，是为了让任何人 clone 仓库后都能复现，而不需要购买商业游戏、训练巨型模型或接触反作弊系统。

## 智能体能力

- 跨五个小局保留遗迹、访问历史、目标置信度和敌方踪迹；会漂移的地形与能量场只信任当前观测；
- 利用地图的反对角线对称性补全可靠信息；
- 通过“本回合得分变化 + 不同单位所处位置”反推出遗迹周围真正的隐藏得分格；
- 动态分配侦察、探测、占点、拦截和回能角色；
- 用带风险代价的 A* 绕开小行星，并权衡未知区域、星云和负能量格；
- 只对堆叠敌人、低能量敌人或争夺目标的敌人使用范围攻击，减少无效消耗。

## 三分钟运行

需要 Python 3.11–3.13，推荐使用 [`uv`](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/bihraint-oss/cortex-arena.git
cd cortex-arena
uv sync --locked --extra dev
uv run cortex-arena doctor
uv run cortex-arena play --seed 42 --output replays/demo.html
```

在浏览器打开 `replays/demo.html` 就能观看整场对局。也可以打开实时窗口：

```bash
uv run cortex-arena play --seed 42 --render
```

## 批量评测

评测命令会交替双方出生点、固定并递增随机种子，然后输出逐局结果和汇总 JSON：

```bash
uv run cortex-arena benchmark --games 10 --seed 100
```

项目自带的对手只是透明、确定性的冒烟测试基线，不代表排行榜水平。正式比较前请阅读 [评测说明](docs/evaluation.md)。

v0.1.0 已验证结果：在 `luxai-s3==0.2.1` 中使用种子 100–109、每局交替出生方，取得 **10 胜 0 负、总小局 44:6**；Apple Silicon + Python 3.13.12 用时 48.8 秒。原始记录见 [`reports/baseline-v0.1.0.json`](reports/baseline-v0.1.0.json)。

## 安全边界

本项目只控制本地、开源的 Lux 模拟环境，不截取任意桌面窗口，不模拟商业平台键鼠输入，不读游戏内存，不绕过反作弊，也不用于在线排位或奖励获取。后续计划接入的是 0 A.D. 官方提供的离线 HTTP 强化学习接口，而不是 Steam 自动化。

原创代码使用 [MIT License](LICENSE)。Lux AI S3 环境单独以 Apache-2.0 发布，未复制进本仓库；详情见 [NOTICE](NOTICE)。
