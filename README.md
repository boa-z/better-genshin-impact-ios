# bgi-touch — BetterGI 跨平台移植（iPhone 触控版）

把 [better-genshin-impact](https://github.com/babalae/better-genshin-impact)（BetterGI，Windows 专用）的自动化能力移植为 **macOS / Linux 可运行**的 Python 项目：通过 [DeviceHub Mask] 的 MCP 服务（`http://127.0.0.1:8009/mcp`）控制 **iPhone 上的原神**，并支持把 [bettergi-scripts-list](https://github.com/babalae/bettergi-scripts-list) 的社区脚本转换/直接运行。

## 技术路线

```
bettergi-scripts-list 脚本 ──bgi-touch convert──▶ 可运行脚本 + 兼容报告
                                                    │
┌───────────────────────────────────────────────────▼──────────────┐
│ engine   JS 兼容运行时（pythonmonkey/SpiderMonkey，bettergi.d.ts │
│          API 表面）· 战斗 DSL 执行器 · 键鼠宏回放 · pathing 框架 │
├──────────────────────────────────────────────────────────────────┤
│ vision   OpenCV 模板匹配 · RapidOCR（PaddleOCR 模型，与原版同源）│
│          1920×1080 脚本坐标系 ↔ iPhone 屏幕的贴边锚点映射        │
├──────────────────────────────────────────────────────────────────┤
│ input    键鼠→触控：WASD→虚拟摇杆（multi_touch 手势泵）          │
│          E/Q/Space→按钮点按 · 鼠标相对移动→视角区滑动            │
│          数字键切人→按活跃槽位换算队伍行号                       │
├──────────────────────────────────────────────────────────────────┤
│ device   官方 mcp SDK（Streamable HTTP）→ devicehub-mask         │
│          同步外观（后台事件循环线程），横竖屏坐标空间自适应      │
└──────────────────────────────────────────────────────────────────┘
```

关键设计（与原版语义对齐）：

- **同步语义**：原版脚本是 ClearScript 绑定同步 C# 方法的调用风格。本项目所有 API 均为阻塞式，`await sleep()` 在阻塞后立即 resolve，时序一致。
- **手势泵**：devicehub-mask 触控手势原子且 ≤5s，"按住 W"由后台线程连发 multi_touch 维持；W+Shift 等组合合并进同一手势的多个触点。
- **坐标系**：脚本/模板资产按 1920×1080 基准；按高度等比缩放 + 依元素所在三分位做左/中/右锚点重定位（原神移动端 HUD 贴边锚定，iPhone 19.5:9 比 16:9 宽）。
- **朝向自适应**（真机实测）：游戏横屏时截图流仍是竖屏帧（内容旋转 90°），而 tap 坐标空间跟随 `status.orientation` 动态变化。截图按帧宽高自动旋转；点按映射按 status 动态启停。

## DeviceHub profile

默认读取 DeviceHub 的 `Genshin-Impact-fixed-16by9`，按住状态通过
`start_game_session` / `set_game_input` 以租约维持；profile 不可用时自动回退旧手势泵。
也可以用 `--keymap-profile-file "$HOME/Library/Application Support/com.devicehub.mask/profiles/Genshin-Impact-fixed-16by9.json"`
从本地 v2 JSON 读取，或用 `--no-keymap-profile` 强制使用手势泵。

`status.screen_size` 可能是视频流缩略尺寸，坐标变换以原生 screenshot 帧为准；
iPhone 13 Pro Max 横屏逻辑空间约为 `2778x1284`。

DeviceHub MCP 与 headless 配置在 `config/devicehub.json`。将
`headless.executable` 改为 `devicehub-headless` 的绝对路径后，MCP 不可用时程序会
按配置自动启动 headless；可同时设置 `workingDirectory`、`args`、启动超时和
`shutdownOnExit`。相对路径以该配置文件所在目录为基准。

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -e .                       # 核心：mcp + opencv + numpy
.venv/bin/pip install -e '.[ocr,js,models]'      # 推荐：OCR + JS + BetterGI ONNX 模型运行时
.venv/bin/python tools/fetch_map_assets.py --models --tcg --quick-buy --auto-boss  # 地图、模型与任务识别资产
```

前置：macOS/Linux 上运行 DeviceHub Mask 并连接 iPhone（**建议 USB**，Wi-Fi 通道输入注入不稳定），iPhone 已安装原神。

## 使用

**完整使用指南（环境准备/校准/脚本转换与运行/Python API/排障）见 [docs/USAGE.md](docs/USAGE.md)。** 速查：

```bash
bgi-touch status                       # 设备/游戏状态
bgi-touch launch                       # 启动原神
bgi-touch close-game                   # 测试后停止/后台挂起原神
bgi-touch screenshot -o shot.png       # 截图（自动转横屏）
bgi-touch calibrate -o cal.png         # 输出布局标注图，用于校准触控坐标
bgi-touch convert <脚本路径>... -o scripts   # 转换社区脚本（js 包/pathing/键鼠宏/战斗txt）
bgi-touch run scripts/js/<脚本名>       # 运行 JS 脚本包（BetterGI 兼容）
bgi-touch combat 万能战斗策略.txt        # 执行战斗策略 DSL
bgi-touch task AutoCook                 # 执行 BetterGI SoloTask（可用 --config 传 JSON）
bgi-touch task AutoAlbum --config '{"musicLevel":"传说"}'  # 在主题专辑页完成未演奏曲目
bgi-touch task QuickSereniteaPot       # 从背包部署并进入/离开尘歌壶
bgi-touch task UseRedemptionCode --config '{"codes":["CODE1","CODE2"]}'
bgi-touch task AutoArtifactSalvage --config '{"star":4}' # 默认只选择并停在复查页
bgi-touch task CountInventoryItem --config '{"gridScreenName":"Materials","itemNames":["萃凝晶","白铁块"]}'
bgi-touch task CharacterDevelopment --config '{"characterName":"钟离","categories":"属性;武器;天赋"}'
bgi-touch task AutoDomain --config '{"domainRoundNum":2,"rewardRecognitionEnabled":true}'
bgi-touch task OneDragon --config-file '/path/to/User/OneDragon/日常.json'
bgi-touch task AutoFishing --config '{"autoThrowRodEnabled":true,"targetCatches":5}'
bgi-touch macro 宏.json                 # 回放键鼠宏（自动翻译为触控）
bgi-touch pathing 路线.json --dry-run   # 校验并解析 pathing 文件
bgi-touch pathing 路线.json             # 启动地图追踪（需要地图资产与真机）
bgi-touch web                          # WebUI 控制台（实况画面/点按/脚本管理）
```

- 触控布局：`config/controls/genshin-default.json`（已按 iPhone 13 Pro Max 实测校准；其他机型先跑 `calibrate` 对照调整）。
- 原生 UI 键位覆盖：`config/controls/genshin-native-ui.json` 继承默认布局，把 `SPACE` 切到 profile 原始 `Space`；钓鱼/烹饪等小游戏可用 `--layout` 选择。
- 队伍映射：`config/party.json`，如 `{"钟离": 1, "那维莱特": 2}`，供战斗 DSL 切人。
- 脚本 settings：脚本目录放 `user-settings.json` 或 `bgi-touch run --set key=value`。

## 真机收尾

测试结束执行 `bgi-touch close-game`。App Store 版原神可能拒绝 MCP 的 `stop_app`，
此时命令会退回 Home 将其移出前台并挂起；不要让游戏留在前台持续消耗设备性能。

## 移植进度

| 能力 | 状态 |
|---|---|
| 设备控制（截图/点按/滑动/多点触控/启停 App） | ✅ 真机验证 |
| 键鼠→触控映射（摇杆/按钮/视角/切人） | ✅ 真机验证（摇杆前进、跳跃、视角滑动） |
| 模板匹配 / OCR（RapidOCR=PaddleOCR 模型） | ✅ |
| JS 兼容运行时（sleep/log/settings/file/http/识别/键鼠 API） | ✅ 自测通过 |
| 战斗策略 DSL（combat txt / action_params） | ✅ 解析+执行 |
| 键鼠宏 → 触控时间线转换与回放 | ✅ |
| 脚本转换器 + 兼容性报告（COMPAT.md） | ✅ |
| WebUI 控制台（实况预览/手动控制/脚本运行/日志） | ✅ 真机验证 |
| 小地图 SIFT 定位（官方特征库、局部/全局回退、跳点过滤） | ✅ 离线验证 |
| 大地图传送 genshin.tp（SIFT 比例自适应拖动+OCR确认） | ⚠️ 已实现，待真机回归验证 |
| genshin.moveMapTo / tpToStatueOfTheSeven / 大地图缩放 | ⚠️ 已实现，缩放为触控 pinch 近似 |
| pathing 执行（定位+走点+传送+动作+异常重试） | ⚠️ 离线全链路，待真机路线实测 |
| 实时触发器（AutoPick OCR拾取 / AutoSkip 剧情推进） | ⚠️ 已实现，待真机调阈值 |
| 战斗增强（OCR按名切人/技能就绪/敌血条结束检测） | ⚠️ 已实现，待真机调阈值 |
| genshin.returnMainUi / chooseTalkOption / relogin / uid / getPositionFromMap | ✅（部分启发式） |
| SoloTask：AutoFight / AutoWood / AutoDomain | ⚠️ 已实现；AutoDomain 支持 ItemV2 多页奖励识别，待真机验证 |
| SoloTask：AutoCook / AutoFishing / AutoOpenChest | ⚠️ AutoFishing 已含找鱼、选饵、HutaoFisher 抛竿、提竿与拉条闭环；需真机回归 |
| SoloTask：AutoAlbum / AutoEat / AutoMusicGame | ⚠️ 已迁移，需在对应游戏界面真机回归 |
| SoloTask：AutoBoss / AutoLeyLine / AutoStygian / AutoGeniusInvokation | ⚠️ Boss/地脉内置官方路线与树脂策略；幽境已迁移活动导航与领奖状态机；七圣召唤需策略 |
| 快捷任务：尘歌壶 / 一键领取 / 快速购买 / 兑换码 / 邮件奖励 | ⚠️ 已迁移，待对应界面真机回归 |
| SoloTask：AutoArtifactSalvage | ⚠️ 低星快速选择与五星规则筛选已迁移；最终分解有显式安全开关 |
| 背包网格：CountInventoryItem / GetGridIcons / 数量 OCR 对比 | ⚠️ 已迁移，详情 OCR 模式待真机回归 |
| characterDevelopmentTask / CharacterDevelopment | ⚠️ 角色卡、等级、武器、三战斗天赋流程已迁移，待真机回归 |
| 一条龙 OneDragonFlowConfig | ⚠️ 顺序、重复任务 ID、NextTaskId、内置任务与安全完成动作已迁移，待真机回归 |

## 已知约束

- 游戏必须前台横屏；若点按无反应，先 `bgi-touch status` 看 `orientation` 是否为 landscape（服务器感知有延迟，重开 App 或重连可刷新）。
- iOS 系统弹层（游戏模式横幅、控制中心）会遮挡/吞掉触控，脚本长跑时建议开启勿扰。
- 未安装 WDA 也可用（本项目不依赖无障碍树，纯像素方案——原神是 Unity 渲染，WDA 也看不到游戏 UI）。
