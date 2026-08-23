# 路线图

## 已完成 — DeviceHub/iOS 输入与核心运行时

- DeviceHub Mask MCP 设备连接、原生截图、横竖屏坐标变换和游戏 session 租约。
- `Genshin-Impact-fixed-16by9` profile 的原始 KeyboardEvent.code 映射，以及
  `config/controls/genshin-native-ui.json` 的小游戏键位覆盖。
- BetterGI 兼容 JS 运行时、统一 `TaskDispatcher`、战斗 DSL、宏和 WebUI/CLI 任务入口。
- AutoFight、AutoWood、AutoDomain、AutoCook、AutoFishing（鱼条控制）和
-  AutoOpenChest、AutoEat、AutoMusicGame 和 AutoAlbum 的可测试 Python 实现。
- AutoBoss、AutoLeyLineOutcrop、AutoStygianOnslaught 和 AutoGeniusInvokation 的
  路线/策略驱动迁移，以及 `dispatcher.runTask` / `runAuto*Task` 入口兼容。
- QuickSereniteaPot、QuickClaimReward、UseRedemptionCode 和邮件奖励领取的触控流程；
  奖励列表滚动使用触控上滑，文本输入使用 DeviceHub 原生输入。
- AutoArtifactSalvage 的 1~4 星快速选择、4x9 五星网格扫描、详情 OCR 和限时
  JavaScript `Output` 规则；最终分解默认关闭，选择完成后保留人工复查界面。
- 通用背包网格轮廓检测、翻页、详情名称 OCR 和数量区域预处理，以及脚本所需的
  CountInventoryItem、常规分类 GetGridIcons、InventoryCountComparison。

## 已完成 — 地图追踪离线核心

- 对齐 BetterGI pathing JSON 的 `moveMode`/`pointExtParams`/异常处理和实时触发器配置。
- 小地图定位支持短缓存、局部优先、全局回退和异常跳点过滤；连续丢失时会重置匹配状态。
- 地图追踪支持传送、方位点、走点反馈、卡死脱困、失败重试和常用采集/战斗动作。
- CLI/WebUI/JS 入口共用 `PathingExecutor`，样例路线可以直接 `--dry-run` 校验。

## P0 — pathing 真机稳定性

离线链路已可用，剩余工作是不同地图/设备上的真机调参：

1. **小地图定位**（`bgi_touch/pathing/positioner.py`）
   - 小地图裁剪 → SIFT/模板匹配到大地图，特征数据来自原版地图资产。
   - 当前支持短缓存、上次位置附近的局部搜索、全局回退和跳点过滤；相机朝向检测仍为简化实现。
2. **大地图传送**（`PathingExecutor._teleport`）
   - 打开地图 → 按世界坐标拖动地图 → 点传送锚点 → OCR/模板确认。
   - `genshin.moveMapTo`、`set/getBigMapZoomLevel` 和可见七天神像传送已接入；
     pinch 缩放仍需在真机上校准等级与手势增益。

3. **真机路线回归**
   - 优先验证短路线和包含 `teleport`、`target`、`combat_script`、`nahida_collect`、
     `mining` 的路线；确认完成后执行 `bgi-touch close-game` 挂起原神。

## P1 — 实时触发器与任务视觉回归

AutoPick、AutoSkip 已有后台截图循环；需要在更多 iOS HUD 缩放下校准模板阈值。
AutoCook 的稳定峰值/下降检测、AutoFishing 的鱼条轮廓控制和 AutoOpenChest 的
模板导航已经迁移，真机需要分别从烹饪、钓鱼、宝箱交互状态启动回归。

## P1 — 战斗增强

- `check`（战斗结束检测）：原版用经验条/YOLO；当前使用敌血条启发式。
- `ready`（技能就绪检测）：技能图标区域亮度/颜色判断。
- 队伍识别：右侧角色名 OCR → 自动生成 party_slots，替代手工 `config/party.json`。

## P2 — 原生 SoloTask 扩展

已完成路线/策略驱动的 AutoBoss、AutoLeyLineOutcrop、AutoStygianOnslaught、
AutoGeniusInvokation 和 AutoAlbum 迁移；当前真机回归仍需分别准备活动入口、战斗
策略、七圣召唤策略和主题专辑界面。

下一批未迁移的上游独立任务是 CharacterDevelopment 和部分一条龙状态机。它们依赖复杂的网格识别
或大量页面状态，需要继续移植对应识别器，不能只用通用 OCR 点击流代替。

## 工程

- 单元测试（识别层用录制帧回放，无需真机）
- 多机型布局 profile（iPad、不同 iPhone）与 HUD 自动校准（模板匹配按钮图标）
- OCR 后端可换为 onnxruntime + 原版同款 PP-OCR 模型文件以进一步对齐
