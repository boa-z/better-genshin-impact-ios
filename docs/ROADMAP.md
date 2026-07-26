# 路线图

## P0 — pathing 可用化（最大缺口）

pathing 脚本（社区仓库 5000+ 个）依赖两件事，均已留好接口：

1. **小地图定位**（`bgi_touch/pathing/executor.py` 的 `Positioner` 协议）
   - 原版方案：小地图裁剪 → SIFT/模板匹配到大地图，特征数据来自 NuGet 包
     `BetterGI.Assets.Map`（不在脚本仓库内）。
   - 移植路径：下载/提取原版地图资产 → 用 OpenCV SIFT 复刻
     `SceneBaseMap.GetMiniMapPosition`（先局部搜索上次位置附近，失败再全局）。
   - 相机朝向检测（小地图视野扇形）已实现简化版：`camera_orientation_deg`。
2. **大地图传送**（`PathingExecutor._teleport`）
   - 打开地图 → 缩放到合适级别 → 按世界坐标拖动地图 → 点传送锚点 → 确认。
   - 依赖大地图定位（同上）+ 地图缩放/拖动的像素-世界坐标换算常数。

## P1 — 实时触发器

原版 ~50ms 帧循环驱动 AutoPick（自动拾取 F）、AutoSkip（自动剧情）。
移植：后台线程 1-2 fps 截图 → 模板/OCR 检测 → 触发点按。需要从原版仓库
`GameTask/*/Assets/` 提取模板 PNG（注意按 1080p 基准，识别层会自动缩放）。

## P1 — 战斗增强

- `check`（战斗结束检测）：原版用经验条/YOLO；可先用"屏幕无敌血条"启发式。
- `ready`（技能就绪检测）：技能图标区域亮度/颜色判断。
- 队伍识别：右侧角色名 OCR → 自动生成 party_slots，替代手工 config/party.json。

## P2 — 原生 SoloTask

AutoFight / AutoDomain / AutoWood 等按需移植；多数由"识别 + 战斗 DSL + 点击流"
组合而成，地基（识别层/输入层/DSL）已就绪。

## 工程

- 单元测试（识别层用录制帧回放，无需真机）
- 多机型布局 profile（iPad、不同 iPhone）与 HUD 自动校准（模板匹配按钮图标）
- OCR 后端可换为 onnxruntime + 原版同款 PP-OCR 模型文件以进一步对齐
