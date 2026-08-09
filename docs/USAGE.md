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

### 1.4 配置 DeviceHub headless

项目默认读取 `config/devicehub.json`。配置 headless 可执行文件后，CLI、WebUI 和
脚本运行入口在 MCP 尚未启动时会自动启动它：

```json
{
  "mcpUrl": "http://127.0.0.1:8009/mcp",
  "headless": {
    "executable": "/opt/devicehub/devicehub-headless",
    "workingDirectory": "/opt/devicehub",
    "args": [],
    "autoStart": true,
    "startupTimeoutSeconds": 20,
    "shutdownOnExit": true
  }
}
```

`executable` 也可以写为相对于 `config/devicehub.json` 的路径；省略
`--mcp-listen` 时程序会根据 `mcpUrl` 自动补上对应监听地址。命令行
`--url`、环境变量 `BGI_MCP_URL` 的优先级高于配置文件；配置文件路径可由
`--devicehub-config` 或 `BGI_DEVICEHUB_CONFIG` 覆盖。headless 归档中的
`devicehub-headless`、`dist/` 和 sidecar 应保持原有相对位置，工作目录通常设为
归档顶层目录。配置完成后不再需要手工先启动 headless。

### 1.5 DeviceHub 键位 profile

默认读取 DeviceHub Mask 的 `Genshin-Impact-fixed-16by9` profile。程序启动时会先建立
active device session，再读取 profile；WASD、技能和切人等按键优先通过
`start_game_session` + `set_game_input` 发送，租约过期会自动释放输入。服务器不支持
这些工具时会自动回退到 `multi_touch` 手势泵。

也可以直接指定用户目录中的 v2 JSON：

```bash
bgi-touch --keymap-profile-file "$HOME/Library/Application Support/com.devicehub.mask/profiles/Genshin-Impact-fixed-16by9.json" status
bgi-touch --no-keymap-profile status
```

profile 使用浏览器 `KeyboardEvent.code`（例如 `KeyW`、`Digit1`）。布局文件中的
`profileCode` 保持 BetterGI 的 PC 语义；本 profile 将 `Space` 放在冲刺键、
`ShiftLeft` 放在跳跃键，因此配置中已显式交换这两个 profile code。

布局支持 `extends` 覆盖，不需要复制整份布局。原生 UI 小游戏可使用：

```bash
bgi-touch --layout config/controls/genshin-native-ui.json task AutoFishing \
  --config '{"targetCatches":1}'
bgi-touch --layout config/controls/genshin-native-ui.json task AutoCook
```

默认战斗布局仍使用 `config/controls/genshin-default.json`；`--layout` 只改变
BetterGI 语义到 profile 原始键码的本地解释，不会修改 DeviceHub 的原始 profile。

---

## 2. 快速开始

```bash
# ① 确认设备连接与坐标变换
bgi-touch status
# 关注输出里的 device.status == "connected"、transform（iPhone 13 Pro Max 约 2778x1284）

# ② 启动原神并等它进到主界面（首次登录/公告需手动点掉）
bgi-touch launch

# ③ 截图确认画面（自动把竖屏帧转成横屏）
bgi-touch screenshot -o shot.png

# ④ 校准触控布局（首次使用/换机型必做，见第 3 节）
bgi-touch calibrate -o cal.png
```

调试或真机测试结束后执行：

```bash
bgi-touch close-game
```

App Store 版原神可能拒绝 MCP 的 `stop_app`，命令会退回 Home 将其移出前台并挂起；
此时 `app_status` 仍可能显示 running，但进程应不再占用游戏渲染资源。

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
要调整的是 `buttons` 里的坐标。启用 DeviceHub profile 时，按键坐标由 profile
映射执行，`buttons` 仍用于视角/鼠标参考坐标和 profile 失效时的回退路径。

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

# 原生 SoloTask：任务参数可用 --config 或 --config-file
bgi-touch task AutoDomain --config '{"domainRoundNum":1}'
bgi-touch task AutoOpenChest --config '{"timeoutSeconds":60}'
# 在千音雅集的国家主题专辑页运行；musicLevel 也支持“所有”/normal/master 等别名
bgi-touch task AutoAlbum --config '{"musicLevel":"传说","songCount":13}'

# 键鼠宏：直接给原始宏 JSON，运行时自动翻译为触控
bgi-touch macro ~/dev/bettergi-scripts-list/repo/js/AutoCrystalfly/assets/枫丹-塔拉塔海谷.json

# pathing：--dry-run 会校验 BetterGI 路线字段并输出统计
bgi-touch pathing scripts/pathing/01-孑遗的留迹x2.json --dry-run

# 地图追踪：需要 assets/map/Teyvat 下的官方 SIFT 特征库
bgi-touch pathing scripts/pathing/01-孑遗的留迹x2.json
```

地图追踪会按小地图 SIFT 的局部结果优先、全局结果回退策略更新坐标；连续跳点或
丢失定位时会重置匹配缓存并重试。路线 JSON 中的 `teleport`、`orientation`、
`path`、`target`、`move_mode`、`point_ext_params` 和 `config.realtime_triggers`
会被保留。当前可执行的常用路点动作包括 `combat_script`、`fight`、`mining`、
`normal_attack`、`elemental_skill`、`nahida_collect`、元素采集、`pick_around`、
`pick_up_collect`、`fishing`、`use_gadget`、`stop_flying`、`up_down_grab_leaf`、
`log_output` 和 `exit_and_relogin`。

Windows 专属的队伍自动识别、低血量回血、设置游戏时间和千星奇域流程不会在 iOS
侧自动模拟；遇到这些动作会写入日志并继续保留路线控制权。`exit_and_relogin`
会复用 iOS 侧的重登流程，执行前应确认账号已登录且允许较长等待。真机长路线完成后
请执行 `bgi-touch close-game`，App Store 版无法强杀时命令会退回 Home 挂起原神。

`AutoAlbum` 需要提前打开千音雅集的国家主题专辑页，不能从“全部歌曲”页面启动。
默认处理传说难度的 13 首曲目；`musicLevel` 可设为“普通”“困难”“大师”“传说”或
“所有”，`mustCanorusLevel` 可改为只跳过已经获得“大音天籁”的曲目。每首歌结束后
程序通过列表按钮回到专辑页，再点击原版相同的下一曲位置。

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
  拖拽 = 滑动**；可叠加触控布局标注（等价 `calibrate`）辅助校准。预览优先读取
  最近一次任务/触发器截图，自动化运行期间不会额外抢占截图请求；右侧帧龄用于判断
  预览是否为缓存帧。
- **手动控制**：WASD 按住式移动（按下=key_down，松开=key_up）、普攻/E/Q/跳/
  冲刺/F/菜单、1-4 切人、视角四方向。
- **脚本面板**：列出 `scripts/` 下已转换的四类脚本及 BetterGI SoloTask（JS 包显示兼容结论），
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
task <名称> [--config JSON]    执行 BetterGI SoloTask
run <脚本目录> [--set k=v]   运行 JS 脚本包
combat <文件.txt>            执行战斗策略 DSL
macro <宏.json>              回放键鼠宏（自动翻译为触控）
pathing <文件.json> [--dry-run]  解析/执行 pathing 路线
web [--host H] [--port P]    启动 WebUI 控制台（默认 127.0.0.1:8899）
```
