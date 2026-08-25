# 路线图

## 已完成 — DeviceHub/iOS 输入与核心运行时

- DeviceHub Mask MCP 设备连接、原生截图、横竖屏坐标变换和游戏 session 租约。
- `Genshin-Impact-fixed-16by9` profile 的原始 KeyboardEvent.code 映射，以及
  `config/controls/genshin-native-ui.json` 的小游戏键位覆盖。
- BetterGI 兼容 JS 运行时、统一 `TaskDispatcher`、战斗 DSL、宏和 WebUI/CLI 任务入口。
- BvLocator 已复用统一识别层支持 `TemplateMatch`、`Ocr`、`OcrMatch`、`ColorMatch` 和
  `ColorRangeAndOcr`，颜色识别不会再因宿主类型被错误拒绝。
- AutoFight、AutoWood、AutoDomain、AutoCook、AutoFishing（主动任务与实时触发器鱼条控制）、
  AutoOpenChest、AutoEat、AutoMusicGame 和 AutoAlbum 的可测试 Python 实现。
- AutoBoss、AutoLeyLineOutcrop、AutoStygianOnslaught 和 AutoGeniusInvokation 的
  专用流程迁移，以及 `dispatcher.runTask` / `runAuto*Task` 入口兼容。
- QuickSereniteaPot、尘歌壶奖励（进壶寻找阿圆、领取好感/宝钱、按配置购买）、
  QuickClaimReward、UseRedemptionCode，以及纪行/邮件奖励领取的触控状态机；
  奖励列表滚动使用触控上滑，文本输入使用 DeviceHub 原生输入。
- AutoArtifactSalvage 的 1~4 星快速选择、4x9 五星网格扫描、详情 OCR 和限时
  JavaScript `Output` 规则；最终分解默认关闭，选择完成后保留人工复查界面。
- 通用背包网格轮廓检测、翻页、详情名称 OCR 和数量区域预处理，以及脚本所需的
  CountInventoryItem、常规分类 GetGridIcons、InventoryCountComparison。
- 背包、圣遗物分解和自动装备乐器等界面型任务会独占共享实时触发器截图/输入通道，
  页面切换期间不会让 AutoPick/AutoSkip 抢先消费过渡帧；暂停前尚未启动的触发器列表也会恢复。
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
- ShellTask 的 macOS/Linux 跨平台命令、超时、取消、输出和不等待语义；任意主机命令
  默认由项目安全配置禁用。
- ScriptGroup 配置组兼容原版 Javascript、KeyMouse、Pathing、Shell 项目，支持禁用状态、
  RunNum、禁止时段、多日周期、三种成功记录跳过策略、失败继续/停止，以及原子
  TaskProgress；中断后从未完成或失败项目恢复。
- 自由演奏 MusicPlayer 支持原琴 JSON、MIDI JSON、网络键谱、21 键和弦触控、移调、
  顺序/单曲循环/随机、自定义 BPM，以及按背包详情 OCR 自动装备乐器。
- BetterGI `notification.send/error` 已接入 Gotify，兼容 JS 通知授权、事件订阅、优先级
  和环境变量 Token；有界后台队列不触碰 DeviceHub 截图器。
- HTML 遮罩已按脚本目录生成稳定存储命名空间，并保护不同脚本使用同名窗口时的消息/资源隔离；
  WebUI 仍通过单一长轮询桥接，不增加截图请求。
- `CheckRewardsTask` 已迁移：打开冒险之证委托页，识别每日奖励领取状态，并通过
  `daily.reward` 事件发送成功/未领取通知；一条龙完成阶段会自动执行该检查。
- `ClaimEncounterPointsRewardsTask` 已迁移：按上游流程回主界面、打开冒险之证、
  定位委托页，支持领取/已领取结果，并在任务结束后安全返回主界面；API 与
  dispatcher 入口共用同一套触控/OCR 流程。
- `GoToAdventurersGuildTask` 已迁移：支持国家路线、好感队伍、`onlyDoOnce`、
  每日委托奖励确认、重新进入凯瑟琳对话和探索任务一键重派；OneDragon 不再重复
  调用历练点领取。真机仍需在已解锁的协会路线和对应语言界面回归按钮阈值。
- BetterGI Common Job 的 `WalkToFTask`、`ScanPickTask` 和
  `LowerHeadThenWalkToTask` 已迁移到共享 iOS 任务层；扫描默认复用 AutoPick 的
  共享截图循环，并通过 dispatcher/JS 入口提供脚本兼容。
- BetterGI `genshin` 公共 Job 的奖励领取、协会/合成台路线、材料合成、调时、重登和
  对话选项已接入统一 dispatcher/JS 入口；菜单 OCR 在实时触发器运行时优先消费缓存帧，
  不再为每次轮询启动第二个 DeviceHub 截图生产者。纪行任务已对齐“纪行点数→奖励页”
  两阶段领取、原石弹窗关闭和手动选择奖励保护；邮件任务支持模板/OCR 双路径、无奖励
 结果和统一返回主界面清理。合成台任务已对齐两次路线尝试、`AutoRunEnabled` 国家差异、
 交互失败后的二次 F/后退一步/F 责任链、最后对话选项和合成页进入确认；浓缩树脂会
 读取原粹/浓缩数量，按 `minResinToKeep` 与 5 个库存上限计算次数，并执行数量按钮、
  白色确认、黑色确认和安全返回主界面。材料合成已接入 ItemV2 网格选材、材料类型
  CSV 回退、数量滑块校准/校验、二次确认和产物结果识别。
- QuickBuy 的普通商店与尘歌壶商店数量滑块、购买确认分支；货币模板由资产下载器
  固定版本获取，缺少模板时要求显式指定商店类型以避免误购。
- AutoBoss 的 41 个官方首领路线分型、树脂耗尽/指定次数策略、须臾与脆弱树脂补充、
  征讨之花模板导航、领奖结果汇总、死亡重试、回神像和战后二次定位流程。
- AutoLeyLineOutcrop 的 269 个地脉节点、378 条图边和 632 份官方路线，支持大地图
  花朵定位、最短传送路线、目标纠偏、树脂耗尽/取小值、领奖树脂优先级和每日一条龙配置。
- AutoStygianOnslaught 的活动菜单、地图传送、入口、困难难度、Boss 1~3、胜负结果、
  地脉花、固定/自动树脂策略和继续/退出状态机；自定义路线仅作为可选入口覆盖。
- LogParse 离线核心：兼容 BetterGI 标准日志文件名、多实例标识、配置组/脚本耗时、拾取物、
  传送重试/脱困/复活/战斗超时/异常统计，并通过不连接设备的 CLI 输出 JSON 或终端摘要。

## 已完成 — 地图追踪离线核心

- 对齐 BetterGI pathing JSON 的 `moveMode`/`pointExtParams`/异常处理和实时触发器配置。
- 提瓦特、层岩巨渊、渊下宫、旧日之海、远古圣山、空之神殿和霜月均已接入各自
  坐标系、官方地图资产与多楼层 SIFT；执行器会按路线 `map_name` 自动切换定位器。
- 小地图定位支持 BetterGI 风格的 212→156 中心裁剪、径向渐晕/图标/背景遮罩，
  再进行短缓存、局部优先、跨楼层全局回退和异常跳点过滤；连续丢失时会重置匹配状态。
- 地图追踪支持传送、方位点、走点反馈、卡死脱困、失败重试和常用采集/战斗动作。
- 路点 `pointExtParams.misidentification` 已执行 `unrecognized/pathTooFar` 的
  `previousDetectedPoint` 与 `mapRecognition` 回退，地图识别期间会暂停实时触发器并在
  返回移动前关闭大地图；`genshin.getPositionFromMapWithMatchingMethod` 的上游重载也已对齐。
- CLI/WebUI/JS 入口共用 `PathingExecutor`，样例路线可以直接 `--dry-run` 校验。
- MapMask 实时触发器复用同一截图循环，后台只保留最新待定位帧；主界面输出玩家世界
  坐标，大地图输出当前视野矩形和楼层。WebUI 只轮询内存快照并裁剪本地地图底图，
  不会创建第二个 DeviceHub 截图生产者。
- 路线 `farming_info` 已接入凌晨 4 点切日的每日统计、米游社已有数据合并和
  精英/小怪上限跳过策略。

## P0 — pathing 真机稳定性

离线链路已可用，剩余工作是不同地图/设备上的真机调参：

1. **小地图定位**（`bgi_touch/pathing/positioner.py`）
   - 小地图裁剪 → BetterGI 预处理遮罩 → SIFT/模板匹配到大地图，特征数据来自原版地图资产。
   - 当前支持短缓存、上次位置附近的局部搜索、全局回退和跳点过滤；相机朝向已移植
     BetterGI 的极坐标边缘峰值算法，并在低置信度时回退兼容检测。
2. **大地图传送**（`PathingExecutor._teleport`）
   - 打开地图 → OCR 切换目标区域 → 按该地图坐标系拖动 → 点传送锚点 → OCR/模板确认。
   - 已接入上游 `tp.json` 传送点索引；`force=false` 会按最近点语义吸附并选择国家，
     `force=true` 保留原始坐标点击。
   - `genshin.moveMapTo`、`set/getBigMapZoomLevel` 和可见七天神像传送已接入；
     pinch 缩放仍需在真机上校准等级与手势增益。

3. **真机路线回归**
   - 优先验证短路线和包含 `teleport`、`target`、`combat_script`、`nahida_collect`、
     `mining` 的路线；确认完成后执行 `bgi-touch close-game` 挂起原神。
   - AutoTrack 已接入任务距离 OCR、蓝色目标标记转向、到点停止和远距离任务页最近锚点；
     可先用当前账号已有任务验证，不要求地图传送点完整解锁。
   - iPhone 13 Pro Max 已验证渊下宫 → 层岩巨渊 OCR 区域切换、层岩大地图定位、
     `moveMapTo`、传送确认/加载完成和落点小地图定位；其余独立地图仍需逐图回归。

## P1 — 实时触发器与任务视觉回归

AutoPick、AutoSkip 已有后台截图循环；AutoSkip 已对齐自定义/内置暂停与优先选项、每日委托奖励确认、
探索派遣领奖、邀约分支、交互键选择、提交物品状态机、黑屏推进、普通剧情页关闭、底部三角道具页
与初见角色横幅，并保护主界面、大地图、引导札记、聊天记录和且试身手。iOS 没有桌面进程音频回环，
语音等待配置保留兼容入口，宿主提供 `voice_waiter` 时才执行音频结束等待；仍需在更多 iOS HUD 缩放下
校准模板与颜色阈值。
AutoCook 的稳定峰值/下降检测、AutoFishing 的鱼类 YOLO、鱼饵 ItemV2 识别、
HutaoFisher 抛竿距离模型、提竿和鱼条控制，以及 AutoOpenChest 的模板导航已经迁移；
真机需要分别从烹饪点、钓鱼点和宝箱附近启动回归。

## P1 — 战斗增强

- `check` 与 AutoFight 战斗结束检测已支持开战阻断、敌血条可见时有限跳过、按时间或
  角色快速检查、切人后检查，以及通过 DeviceHub `L/X` 打开/关闭队伍页并校验黄条白块；
  仍需真机校准 iOS 队伍页颜色签名；经验值图标模板和颜色二次确认已接入同一帧流，
  可按精英经验结果门控万叶/琴战技聚怪及战后扫描拾取，不启动第二截图生产者。
- `ready`（战斗 HUD 就绪检测）：等待移动端队伍 HUD/主界面可交互；
  `IsSkillReady`/`WaitSkillCd` 单独保留元素战技冷却语义。
- JSON 战斗策略已支持预动作、稳定优先级展开、动作历史条件、技能/队伍条件、
  `EnsureCast` 与每轮单截图缓存；已通过本机社区公式化锄地策略语料解析，视觉阈值待真机校准。
- 队伍识别：右侧角色名 OCR → 自动生成 party_slots，替代手工 `config/party.json`；
  角色重组已按 BetterGI 的 `usePhysicalSlots` 语义识别单机/联机玩家编号，
  将可控逻辑槽位映射到实际物理槽位并同步战斗缓存。

## P2 — 原生 SoloTask 扩展

已完成专用流程驱动的 AutoBoss、AutoLeyLineOutcrop、AutoStygianOnslaught、
AutoGeniusInvokation 和 AutoAlbum 迁移；当前真机回归仍需分别准备有效活动期、战斗
策略、七圣召唤策略和主题专辑界面。

FarmingPlan 已接入路径执行统计、凌晨 4 点切日和米游社数据合并，ScriptGroup/TaskProgress
已形成调度闭环；旅行札记支持 DS 请求签名、角色选择、分页增量、三个月 JSON 缓存、当天
统计和无设备 CLI，自动更新仅在显式提供 `BGI_MIYOUSHE_COOKIE` 时启用且不触碰截图器。
QuickForge 上游当前仍为空类；日志分析已可输出自包含 HTML 表格（含配置组、耗时、拾取
和故障列），现已补齐表头排序、凌晨 4 点分日的旅行札记摩拉统计，以及配置组/脚本摩拉
明细；报告可从本地札记缓存离线生成。下一批继续移植有实际业务实现的专用识别器，不能
只用通用 OCR 点击流代替。

## 工程

- 单元测试（识别层用录制帧回放，无需真机）
- 多机型布局 profile（iPad、不同 iPhone）与 HUD 自动校准（模板匹配按钮图标）
- OCR 后端可换为 onnxruntime + 原版同款 PP-OCR 模型文件以进一步对齐
