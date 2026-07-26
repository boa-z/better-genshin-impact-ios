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

## 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -e .                       # 核心：mcp + opencv + numpy
.venv/bin/pip install -e '.[ocr,js]'             # 推荐：RapidOCR + pythonmonkey
```

前置：macOS/Linux 上运行 DeviceHub Mask 并连接 iPhone（**建议 USB**，Wi-Fi 通道输入注入不稳定），iPhone 已安装原神。

## 使用

**完整使用指南（环境准备/校准/脚本转换与运行/Python API/排障）见 [docs/USAGE.md](docs/USAGE.md)。** 速查：

```bash
bgi-touch status                       # 设备/游戏状态
bgi-touch launch                       # 启动原神
bgi-touch screenshot -o shot.png       # 截图（自动转横屏）
bgi-touch calibrate -o cal.png         # 输出布局标注图，用于校准触控坐标
bgi-touch convert <脚本路径>... -o scripts   # 转换社区脚本（js 包/pathing/键鼠宏/战斗txt）
bgi-touch run scripts/js/<脚本名>       # 运行 JS 脚本包（BetterGI 兼容）
bgi-touch combat 万能战斗策略.txt        # 执行战斗策略 DSL
bgi-touch macro 宏.json                 # 回放键鼠宏（自动翻译为触控）
bgi-touch pathing 路线.json --dry-run   # 解析 pathing 文件
```

- 触控布局：`config/controls/genshin-default.json`（已按 iPhone 13 Pro Max 实测校准；其他机型先跑 `calibrate` 对照调整）。
- 队伍映射：`config/party.json`，如 `{"钟离": 1, "那维莱特": 2}`，供战斗 DSL 切人。
- 脚本 settings：脚本目录放 `user-settings.json` 或 `bgi-touch run --set key=value`。

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
| genshin.returnMainUi / chooseTalkOption / relogin / uid | ⚠️ 启发式实现，待更多实测 |
| 实时触发器（自动拾取/自动剧情） | ❌ 计划中（需帧循环 + 模板资产） |
| pathing 执行（小地图定位/大地图传送） | ❌ 框架就绪，定位待地图资产接入（见 docs/ROADMAP.md） |
| 原生 SoloTask（自动战斗收尾检测/秘境/钓鱼等） | ❌ 计划中 |

## 已知约束

- 游戏必须前台横屏；若点按无反应，先 `bgi-touch status` 看 `orientation` 是否为 landscape（服务器感知有延迟，重开 App 或重连可刷新）。
- iOS 系统弹层（游戏模式横幅、控制中心）会遮挡/吞掉触控，脚本长跑时建议开启勿扰。
- 未安装 WDA 也可用（本项目不依赖无障碍树，纯像素方案——原神是 Unity 渲染，WDA 也看不到游戏 UI）。
