# 使用指南

本文说明如何在 macOS / Linux 上用 bgi-touch 自动化 iPhone 上的原神，
以及如何把 bettergi-scripts-list 的社区脚本转换过来运行。

---

## 1. 环境准备

### 1.1 前置条件

| 项 | 要求 |
|---|---|
| 电脑 | macOS 或 Linux，Python ≥ 3.11 |
| DeviceHub Mask | 已运行，MCP 服务默认在 `http://127.0.0.1:8009/mcp` |
| iPhone | 已连接 DeviceHub Mask，**强烈建议 USB 连接**（Wi-Fi 通道实测会出现"画面正常但触控/按键全部失效"） |
| 原神 | iPhone 已安装（`com.miHoYo.Yuanshen`），账号已登录过 |

不需要安装 WebDriverAgent：本项目是纯像素方案（截图识别 + 触控注入），
原神是 Unity 渲染，无障碍树方案本来也不可用。

### 1.2 安装

```bash
cd better-genshin-impact-ios
python3 -m venv .venv
.venv/bin/pip install -e '.[ocr,js]'
```

可选依赖说明：

- `[ocr]` → RapidOCR（PaddleOCR 模型，与 BetterGI 原版同源）。不装则 OCR
  识别返回空结果，纯模板匹配的脚本仍可用。
- `[js]` → pythonmonkey（SpiderMonkey）。不装则不能 `bgi-touch run` JS 脚本包，
  但战斗 DSL / 键鼠宏 / 转换器不受影响。

安装后 `bgi-touch` 命令在 `.venv/bin/` 下；也可以用
`.venv/bin/python -m bgi_touch.cli` 等价调用。下文示例假设已
`source .venv/bin/activate` 或使用完整路径。

### 1.3 指定 MCP 地址

默认 `http://127.0.0.1:8009/mcp`，两种覆盖方式：

```bash
bgi-touch --url http://192.168.1.10:8009/mcp status   # 命令行参数
export BGI_MCP_URL=http://127.0.0.1:8009/mcp          # 环境变量
```

---

## 2. 快速开始

```bash
# ① 确认设备连接与坐标变换
bgi-touch status
# 关注输出里的 device.status == "connected"、transform（如 2816x1296, scale=1.2）

# ② 启动原神并等它进到主界面（首次登录/公告需手动点掉）
bgi-touch launch

# ③ 截图确认画面（自动把竖屏帧转成横屏）
bgi-touch screenshot -o shot.png

# ④ 校准触控布局（首次使用/换机型必做，见第 3 节）
bgi-touch calibrate -o cal.png
```

---

## 3. 校准触控布局（首次必做）

不同机型分辨率、游戏内 HUD 缩放设置都会移动按钮位置。仓库自带的
`config/controls/genshin-default.json` 按 **iPhone 13 Pro Max（2816×1296）**
实测校准；其他设备按以下流程核对：

1. 进入游戏主界面（能看到摇杆和技能按钮）。
2. 运行 `bgi-touch calibrate -o cal.png`。
3. 打开 `cal.png`：红圈红字 = 配置里各按钮的当前位置；绿圈 = 摇杆中心与半径；
   蓝框 = 视角滑动区域。
4. 对照实际画面，修改 `config/controls/genshin-default.json` 里对应按钮的
   `nx` / `ny`（归一化坐标：`nx = 像素x / 屏宽`，`ny = 像素y / 屏高`）。
5. 重复 2-4 直到红圈落在按钮正中。

验证动作是否生效的最小测试：

```bash
python - <<'EOF'
import time
from bgi_touch.engine.context import GameContext
ctx = GameContext()
ctx.input.key_press("SPACE")   # 应看到角色跳跃
time.sleep(1)
ctx.input.key_down("W"); time.sleep(2); ctx.input.key_up("W")  # 前进 2 秒
ctx.close()
EOF
```

### 布局文件字段

```jsonc
{
  "joystick": { "center": {"nx":0.18,"ny":0.75}, "radiusN": 0.07 },  // 虚拟摇杆
  "camera":   { "region": {...}, "pixelsPerDegree": 6.5 },           // 视角滑动区
  "buttons":  { "attack": {...}, "skill": {...}, ... },              // 各按钮坐标
  "keyMap":   { "W": {"type":"joystick","angleDeg":270},             // PC 键位 → 触控
                "E": {"type":"button","button":"skill"},
                "1": {"type":"party","slot":1} }
}
```

`keyMap` 决定脚本里 `keyPress("E")` 等 PC 键位落到哪个触控动作，一般不用改；
要调整的是 `buttons` 里的坐标。

---

## 4. 队伍配置（战斗切人必需）

移动端没有数字键，切人是点右侧队伍头像行。脚本里的"切到 2 号位"需要知道
角色名与槽位的对应关系，配置在 `config/party.json`：

```json
{"钟离": 1, "那维莱特": 2, "纳西妲": 3, "芙宁娜": 4}
```

槽位顺序 = 游戏内编队顺序。战斗 DSL / JS 脚本里的角色名必须与这里一致，
未配置的角色会跳过切人并在日志告警。

---

## 5. 转换社区脚本

### 5.1 转换命令

```bash
# 单个或批量，自动识别类型；输出默认到 ./scripts/
bgi-touch convert \
  ~/dev/bettergi-scripts-list/repo/js/AutoCrystalfly \
  ~/dev/bettergi-scripts-list/repo/pathing/敌人与魔物/炉壳山鼬/01-孑遗的留迹x2.json \
  ~/dev/bettergi-scripts-list/repo/combat/万能战斗策略（萌新推荐）.txt \
  -o scripts
```

支持的输入与产物：

| 输入 | 识别依据 | 产物 |
|---|---|---|
| JS 脚本包（目录） | 含 `manifest.json` | `scripts/js/<名>/` 完整拷贝 + **COMPAT.md 兼容报告**；内嵌键鼠宏另转出 `.touch.json` 预览 |
| pathing 路线 | JSON 含 `positions` | 校验 + 路点/动作统计 + 拷贝到 `scripts/pathing/` |
| 键鼠宏 | JSON 含 `macroEvents` | 触控时间线 `scripts/keymouse/<名>.touch.json` |
| 战斗策略 | `.txt` | DSL 解析校验 + 拷贝到 `scripts/combat/` |

### 5.2 读懂 COMPAT.md

每个转换后的 JS 包里有 `COMPAT.md`，三种结论：

- ✅ 未发现不兼容 API —— 可直接 `bgi-touch run`
- ⚠️ 部分能力为近似实现 —— 能跑，个别行为与 PC 不同（如 `moveMouseTo`
  在触控端是空操作、实时拾取触发器被忽略）
- ❌ 存在未移植依赖 —— 列出的调用（如 `genshin.tp`、`dispatcher.runTask`）
  执行到会抛错；若它们只在可选分支里，脚本主流程仍可能可用

### 5.3 运行 JS 脚本包

```bash
bgi-touch run scripts/js/RecognitionDemo

# 覆盖脚本设置（对应原版 settings.json 的 UI 配置项）
bgi-touch run scripts/js/AutoCrystalfly --set 循环次数=5 --set 使用队伍=采集队
```

设置的三层优先级：`--set` > 脚本目录下的 `user-settings.json` >
`settings.json` 里的 `default`。

脚本内的相对路径（`assets/...`）沙箱在脚本目录内；`http.request` 仅允许
manifest `http_allowed_urls` 声明过的地址。

### 5.4 执行战斗策略 / 键鼠宏 / pathing

```bash
# 战斗策略：逐行执行（角色名开头的行会先切人，见第 4 节）
bgi-touch combat scripts/combat/万能战斗策略（萌新推荐）.txt

# 键鼠宏：直接给原始宏 JSON，运行时自动翻译为触控
bgi-touch macro ~/dev/bettergi-scripts-list/repo/js/AutoCrystalfly/assets/枫丹-塔拉塔海谷.json

# pathing：--dry-run 只解析统计；实际执行需要地图定位资产（尚未接入，见 ROADMAP）
bgi-touch pathing scripts/pathing/01-孑遗的留迹x2.json --dry-run
```

战斗 DSL 语法速查（与原版一致）：

```
// 注释
钟离 e(hold), wait(0.5), q          // 切到钟离：长按E、等0.5秒、放大招
那维莱特 charge(3), dash            // 重击3秒、冲刺
attack(2), jump, w(0.5)            // 无角色名 = 当前角色：普攻2秒、跳、前进0.5秒
```

支持动作：`e/skill[(hold)]`、`q/burst`、`attack(秒)`、`charge(秒)`、
`dash(秒)`、`jump`、`w/a/s/d(秒)`、`walk(方向,秒)`、`wait(秒)`、`aim`、
`keydown/keyup/keypress(键)`、`moveby(dx,dy)`（转视角）、`ready`、`check`。

---

## 6. WebUI 控制台

```bash
pip install -e '.[web]'          # fastapi + uvicorn
bgi-touch web                    # 默认 http://127.0.0.1:8899
bgi-touch web --host 0.0.0.0 --port 8899   # 局域网访问（注意无鉴权，慎用）
```

功能一览：

- **屏幕实况**：0.5/1/2 fps 可调；**在预览图上点击 = 点按设备对应位置，
  拖拽 = 滑动**；可叠加触控布局标注（等价 `calibrate`）辅助校准。
- **手动控制**：WASD 按住式移动（按下=key_down，松开=key_up）、普攻/E/Q/跳/
  冲刺/F/菜单、1-4 切人、视角四方向。
- **脚本面板**：列出 `scripts/` 下已转换的四类脚本（JS 包显示兼容结论），
  一键运行/停止；同一时刻只允许一个任务，顶部标签显示
  idle/running/done/error/cancelled。
- **转换**：粘贴 bettergi-scripts-list 中的路径，转换产物直接进列表。
- **日志**：任务与操作日志实时滚动（即 CLI 里的输出）。

顶部红色「■ 停止」= 终止当前任务并松开所有按住的触点（跑飞时先按它）。

## 7. 用 Python 直接写脚本

不走 JS 兼容层时，可以直接用本项目的 Python API（同步阻塞式）：

```python
import time
from bgi_touch.engine.context import GameContext
from bgi_touch.engine.recognition import RecognitionObject, Mat

ctx = GameContext()               # 连接设备，读取布局与坐标变换
ctx.launch_game()                 # 启动原神（已在前台则相当于切前台）

# 输入：BetterGI 键鼠语义
ctx.input.key_press("E")          # 元素战技
ctx.input.key_down("W"); time.sleep(2); ctx.input.key_up("W")
ctx.input.move_camera_by(300, 0)  # 右转视角（1080p 像素语义）
ctx.input.click_ref(960, 540)     # 点屏幕中心（1920x1080 参考坐标）
ctx.input.switch_party_slot(2)    # 切 2 号位

# 识别：截图 → 模板匹配 / OCR（坐标均为 1920x1080 参考空间）
region = ctx.capture_region()
ro = RecognitionObject.template_match(Mat.from_file("assets/btn.png"), 0, 0, 960, 540)
hit = region.find(ro)
if hit.is_exist():
    hit.click()
for r in region.find_multi(RecognitionObject.ocr(1200, 300, 700, 700)):
    print(r.text, r.x, r.y)

ctx.close()
```

模板图片请按 1080p 基准截取（识别层会自动按设备比例缩放后匹配）。

---

## 8. 故障排查

| 现象 | 原因与处理 |
|---|---|
| 点按/按键全部无反应，但截图正常 | 大概率 Wi-Fi 连接。改用 USB（DeviceHub Mask 里切换到 USB 通道后重试） |
| 点按位置不对 / 点到别处 | ① `bgi-touch status` 看 `orientation`：游戏横屏但显示 `portrait` 说明服务器还没感知到，重开游戏或重连设备后重试；② 布局未校准，跑 `calibrate` |
| 报错"截到竖屏画面" | 游戏不在前台。先 `bgi-touch launch` |
| 脚本运行到一半没反应 | iOS 弹层（游戏模式横幅、控制中心、通知）会吞触控。长跑前开勿扰模式，别碰设备 |
| OCR 相关识别永远为空 | 未装 OCR 后端：`pip install -e '.[ocr]'` |
| `bgi-touch run` 报 pythonmonkey 缺失 | `pip install -e '.[js]'` |
| JS 脚本抛 `NotImplementedError` | 依赖未移植能力（COMPAT.md 会预警），常见是 `genshin.tp` / `dispatcher.runTask`，见 docs/ROADMAP.md |
| 切人切错角色 | `config/party.json` 与游戏内编队顺序不一致；或脚本中途手动切过人导致内部活跃槽位记录偏移（重启脚本复位） |
| 设备操作偶发超时（CoreDevice timed out） | 设备侧瞬态错误，重试即可；频繁出现则重连设备 |

---

## 9. 命令速查

```
bgi-touch [--url URL] <命令>

status                      设备/游戏状态 + 坐标变换信息
launch                      启动原神
screenshot [-o 文件]         截图（自动转横屏，默认 screenshot.png）
calibrate  [-o 文件]         输出布局标注图（默认 calibrate.png）
convert <路径>... [-o 目录]   转换社区脚本（默认输出 ./scripts）
run <脚本目录> [--set k=v]   运行 JS 脚本包
combat <文件.txt>            执行战斗策略 DSL
macro <宏.json>              回放键鼠宏（自动翻译为触控）
pathing <文件.json> [--dry-run]  解析/执行 pathing 路线
web [--host H] [--port P]    启动 WebUI 控制台（默认 127.0.0.1:8899）
```
