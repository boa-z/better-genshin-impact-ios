# 路线图

## 已完成 — DeviceHub/iOS 输入与核心运行时

- DeviceHub Mask MCP 设备连接、原生截图、横竖屏坐标变换和游戏 session 租约。
- `Genshin-Impact-fixed-16by9` profile 的原始 KeyboardEvent.code 映射，以及
  `config/controls/genshin-native-ui.json` 的小游戏键位覆盖。
- BetterGI 兼容 JS 运行时、统一 `TaskDispatcher`、战斗 DSL、宏和 WebUI/CLI 任务入口。
- AutoFight、AutoWood、AutoDomain、AutoCook、AutoFishing（鱼条控制）、
  AutoOpenChest、AutoEat、AutoMusicGame 和 AutoAlbum 的可测试 Python 实现。
- AutoBoss、AutoLeyLineOutcrop、AutoStygianOnslaught 和 AutoGeniusInvokation 的
  专用流程迁移，以及 `dispatcher.runTask` / `runAuto*Task` 入口兼容。
- QuickSereniteaPot、QuickClaimReward、UseRedemptionCode 和邮件奖励领取的触控流程；
  奖励列表滚动使用触控上滑，文本输入使用 DeviceHub 原生输入。
- AutoArtifactSalvage 的 1~4 星快速选择、4x9 五星网格扫描、详情 OCR 和限时
  JavaScript `Output` 规则；最终分解默认关闭，选择完成后保留人工复查界面。
- 通用背包网格轮廓检测、翻页、详情名称 OCR 和数量区域预处理，以及脚本所需的
  CountInventoryItem、常规分类 GetGridIcons、InventoryCountComparison。
- characterDevelopmentTask 的单/多角色接口、固定尺寸角色卡选择，以及属性、武器、
  普攻/战技/爆发天赋等级读取；角色别名和元素/武器元数据已随项目固定版本保存。
- AutoDomain 奖励页稳定等待、卡片检测、ItemV2 图标匹配、数量 OCR、多页去重与
  BetterGI 名称到累计数量的返回契约；ItemV2 模型由资产下载器按需安装。
- OneDragonFlowConfig 的新旧格式、显式顺序、同名重复任务 ID、NextTaskId 断点、
  内置任务编排与关闭/挂起原神完成动作；自定义配置组可通过 taskConfigs 映射任务。
- AutoGeniusInvokation 的完整策略元数据、投骰与重投、行动骰子、元素调和、回合状态、
  异常状态与阵亡重选；官方 TCG 模板和角色别名元数据可由资产下载器固定版本获取。
- AutoFishing 的全天、白天、夜晚和不调整时间策略；全天模式按 BetterGI 语义依次
  清理 07:00 与 19:00 两个鱼塘阶段。
- JavaScript 参数对象、取消令牌、ServerTime、PostMessage 前台触控近似，以及
  file 文本回调、图片缩放/写入等 bettergi.d.ts 宿主契约。
- QuickBuy 的普通商店与尘歌壶商店数量滑块、购买确认分支；货币模板由资产下载器
  固定版本获取，缺少模板时要求显式指定商店类型以避免误购。
- AutoBoss 的 41 个官方首领路线分型、树脂耗尽/指定次数策略、须臾与脆弱树脂补充、
  征讨之花模板导航、领奖结果汇总、死亡重试、回神像和战后二次定位流程。
- AutoLeyLineOutcrop 的 269 个地脉节点、378 条图边和 632 份官方路线，支持大地图
  花朵定位、最短传送路线、目标纠偏、树脂耗尽/取小值、领奖树脂优先级和每日一条龙配置。
- AutoStygianOnslaught 的活动菜单、地图传送、入口、困难难度、Boss 1~3、胜负结果、
  地脉花、固定/自动树脂策略和继续/退出状态机；自定义路线仅作为可选入口覆盖。

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
AutoCook 的稳定峰值/下降检测、AutoFishing 的鱼类 YOLO、鱼饵 ItemV2 识别、
HutaoFisher 抛竿距离模型、提竿和鱼条控制，以及 AutoOpenChest 的模板导航已经迁移；
真机需要分别从烹饪点、钓鱼点和宝箱附近启动回归。

## P1 — 战斗增强

- `check`（战斗结束检测）：原版用经验条/YOLO；当前使用敌血条启发式。
- `ready`（技能就绪检测）：技能图标区域亮度/颜色判断。
- 队伍识别：右侧角色名 OCR → 自动生成 party_slots，替代手工 `config/party.json`。

## P2 — 原生 SoloTask 扩展

已完成专用流程驱动的 AutoBoss、AutoLeyLineOutcrop、AutoStygianOnslaught、
AutoGeniusInvokation 和 AutoAlbum 迁移；当前真机回归仍需分别准备有效活动期、战斗
策略、七圣召唤策略和主题专辑界面。

下一批迁移重点是 FarmingPlan 等尚未接入统一调度器的页面任务；QuickForge 上游
当前仍为空类。需要继续移植专用识别器，
不能只用通用 OCR 点击流代替。

## 工程

- 单元测试（识别层用录制帧回放，无需真机）
- 多机型布局 profile（iPad、不同 iPhone）与 HUD 自动校准（模板匹配按钮图标）
- OCR 后端可换为 onnxruntime + 原版同款 PP-OCR 模型文件以进一步对齐
