# Environment research and selection

Research was refreshed on 2026-07-22 using project documentation, official repositories, and primary papers.

## Candidates

| Environment | Strength | Main drawback for this repository | Decision |
|---|---|---|---|
| [Lux AI Season 3](https://github.com/Lux-AI-Challenge/Lux-Design-S3) | Real strategy mechanics, partial observation, simultaneous units, browser replay, Apache-2.0 | Competition environment is frozen at `luxai-s3==0.2.1` | **Selected and pinned** |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | Actively maintained, broad game/algorithm coverage, easy macOS wheels | Most games render as boards or text rather than a game-like RTS replay | Strong alternative for search research |
| [0 A.D. R28 RL interface](https://gitea.wildfiregames.com/0ad/0ad/wiki/GettingStartedReinforcementLearning) | Current full RTS with official HTTP reset/step API and nonvisual mode | Requires a separately installed full game and a version-specific experimental client | Planned adapter |
| [PySC2](https://github.com/google-deepmind/pysc2) | Authentic StarCraft II research interface | Large proprietary install, Linux-first testing, and very high full-game training cost | Not suitable for the first release |
| [MicroRTS](https://github.com/Farama-Foundation/MicroRTS) / [MicroRTS-Py](https://github.com/Farama-Foundation/MicroRTS-Py) | Classic affordable RTS research benchmark | Both official repositories were marked deprecated on 2025-08-11 | Rejected as a new foundation |
| [MAgent2](https://github.com/Farama-Foundation/MAgent2) | Large multi-agent battles through PettingZoo | Abstract grid battle and weaker match/replay story than Lux | Useful future benchmark |
| [OpenTTD NoAI](https://wiki.openttd.org/en/Development/Script/) | Stable in-game scripting API and excellent headless support | Transport economy rather than combat strategy; Squirrel-only agent | Good low-risk alternative |

## Why not start with visual desktop automation?

Frameworks such as [Cradle](https://github.com/BAAI-Agents/Cradle) demonstrate a screenshot → planning → keyboard/mouse loop, but their own results show that precise and real-time game control remains fragile. General game models are also less reusable than their marketing may suggest:

- [OpenAI VPT](https://github.com/openai/Video-Pre-Training), [MineRL](https://github.com/minerllabs/minerl), and [Voyager](https://github.com/MineDojo/Voyager) are tightly coupled to Minecraft data or APIs;
- [NitroGen](https://github.com/MineDojo/NitroGen) states that its single-frame controller is weak on keyboard/mouse RTS and MOBA games, and its model license is limited to non-commercial research;
- [GameNGen](https://gamengen.github.io/) is a learned game simulator, not a controller for an external game.

Desktop automation also creates avoidable compliance risk. Current [Steam Subscriber Agreement](https://store.steampowered.com/subscriber_agreement/) language prohibits scripts, bots, macros, and other non-human control systems interacting with Steam content and services. A screenshot-and-virtual-input implementation does not automatically avoid that restriction.

For those reasons Cortex Arena starts with an explicitly exposed, local simulation API. The game process is real, the observations are partial, and the actions are the same actions accepted from every competition agent—but installation, CI, and evaluation remain reproducible.

## Algorithm references

- Santiago Ontañón, [The Combinatorial Multi-Armed Bandit Problem and its Application to Real-Time Strategy Games](https://ojs.aaai.org/index.php/AIIDE/article/view/12626), AIIDE 2013.
- Marc Lanctot et al., [OpenSpiel: A Framework for Reinforcement Learning in Games](https://arxiv.org/abs/1908.09453), 2019.
- Lux AI Challenge authors, [Season 3 game specification](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md).
