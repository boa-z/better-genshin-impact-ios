# bgi-touch 开发指南

BetterGI 的 Python 移植：经 devicehub-mask MCP（默认 `http://127.0.0.1:8009/mcp`）
控制 iPhone 上的原神。**本项目只用 Python**（用户明确要求，勿引入 TS/Node）。

## 常用命令

```bash
.venv/bin/pip install -e '.[ocr,js]'     # 安装（含 RapidOCR 与 pythonmonkey）
.venv/bin/python -m bgi_touch.cli status # CLI 入口 = bgi_touch/cli.py
```

## 结构

- `bgi_touch/device/client.py` — MCP 同步外观（后台 asyncio 线程）。负责 active
  device session、profile/game session、输入租约，以及"游戏横屏但服务器报竖屏"
  时的 tap 空间旋转。
- `bgi_touch/input/layout.py` — 本地布局 + DeviceHub native v2 profile 解析；保留
  profile 原始 `KeyboardEvent.code`，把 BetterGI 规范化键名映射到原始键码。
- `bgi_touch/input/simulator.py` — 键鼠→触控。按住语义靠手势泵线程连发
  multi_touch（旧服务器回退）；可用时优先使用 DeviceHub game session 的 60Hz
  持续输入。切人按活跃槽位换算行号。
- `bgi_touch/vision/` — coordinate.py 是坐标系核心：1920×1080 脚本空间 ↔ 设备
  空间，x 按三分位做左/中/右锚点（移动端 HUD 贴边，19.5:9 ≠ 16:9）。
- `bgi_touch/engine/js_runtime.py` — pythonmonkey 宿主，实现 bettergi.d.ts API。
  全部 API 阻塞式；大小写不敏感靠 JS Proxy（原版 ClearScript 行为）。
  必须在 asyncio.run 里驱动脚本 Promise。
- `bgi_touch/converter/convert.py` — 社区脚本转换 + COMPAT.md 兼容报告。
  新增/移植 API 后同步更新其 SUPPORTED/PARTIAL/UNSUPPORTED 表。

## 真机经验（iPhone 13 Pro Max，勿凭空修改）

- tap 坐标空间跟随 `status.orientation`：landscape 时直接用横屏坐标；portrait
  时需 `P.x = P_W - L.y, P.y = L.x`。截图流朝向独立，按帧宽高判断旋转。
- `status.screen_size` 可能是缩略视频尺寸；`GameContext` 必须以原生截图帧更新
  `ScreenTransform`，iPhone 13 Pro Max 横屏约 2778×1284。
- 输入注入用 **USB** 连接；Wi-Fi 下曾出现视频正常但触控/Home 键全部失效。
- iOS 弹层（游戏模式横幅/控制中心）会吞触控。
- 触控布局在 `config/controls/genshin-default.json`，改前先跑
  `bgi-touch calibrate` 拿标注截图对照。
- 真机调试结束执行 `bgi-touch close-game`；App Store 原神若拒绝 `stop_app`，命令
  会按 Home 后台挂起，确认不要留在前台。

## 约定

- 每完成一个阶段立即 git commit（用户要求），提交信息中文。
- 面向脚本的坐标一律 1920×1080 参考空间；设备像素只在内部出现。
