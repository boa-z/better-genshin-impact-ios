"""AutoDomain SoloTask：自动秘境（对应原版 GameTask/AutoDomain，OCR 驱动版）。

前提：角色已站在秘境内的启动点（可先用 genshin.tp 到秘境并进入）。
每轮流程：启动挑战 → 前进触发战斗 → AutoFight → 挑战达成 → 走向石化古树
→ 使用树脂领奖 → 继续/退出。视觉判定全部 OCR，无 PC 专用模板依赖。

原版的古树寻路用识别转向；此处以「战斗结束点朝向古树直走」近似，
石化古树在多数秘境位于场地正前方，失败时靠交互键容错。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from ..engine.context import GameContext
from ..engine.recognition import Mat, RecognitionObject
from .auto_fight import AutoFightTask


TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "assets" / "templates" / "stygian"
REWARD_RESULT_EXIT = RecognitionObject.template_match(
    Mat.from_file(str(TEMPLATE_ROOT / "exit_button.png")),
    0,
    540,
    960,
    540,
)


class AutoDomainTask:
    def __init__(self, ctx: GameContext, rounds: int = 1,
                 combat_strategy_path: str | None = None,
                 use_condensed_resin: bool = True,
                 reward_recognition_enabled: bool = False,
                 reward_max_pages: int = 3,
                 party_slots: dict[str, int] | None = None,
                 log: Callable[[str], None] = print):
        self.ctx = ctx
        self.rounds = max(1, int(rounds))
        self.use_condensed = use_condensed_resin
        self.reward_recognition_enabled = bool(reward_recognition_enabled)
        self.reward_max_pages = max(1, int(reward_max_pages))
        self.reward_summary: dict[str, int] = {}
        self._reward_recognizer = None
        self.log = log
        self.fight = AutoFightTask(ctx, combat_strategy_path, timeout_s=300,
                                   party_slots=party_slots, log=log)

    # ---- OCR 工具 ----

    def _find_text(self, *texts: str, roi=(0, 0, 1920, 1080)):
        region = self.ctx.capture_region()
        hits = region.find_multi(RecognitionObject.ocr(*roi), limit=25)
        for h in hits:
            if any(t in h.text for t in texts):
                return h
        return None

    def _tap_text(self, *texts: str, roi=(0, 0, 1920, 1080), timeout_s: float = 10) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            h = self._find_text(*texts, roi=roi)
            if h is not None:
                self.log(f"[AutoDomain] 点击「{h.text.strip()}」")
                h.click()
                return True
            self.ctx.sleep(700)
        return False

    def _walk(self, seconds: float) -> None:
        self.ctx.input.key_down("W")
        self.ctx.sleep(seconds * 1000)
        self.ctx.input.key_up("W")

    _tree_model = None

    def _walk_to_tree(self, max_steps: int = 10) -> None:
        """原版 bgi_tree.onnx：检测石化古树，把它转到画面中央后前进。"""
        try:
            if self._tree_model is None:
                from ..vision.yolo import YoloPredictor
                self._tree_model = YoloPredictor("bgi_tree")
        except FileNotFoundError:
            self.log("[AutoDomain] 无古树模型，直走回退")
            self._walk(4.0)
            return
        w = self.ctx.transform.device_width
        for _ in range(max_steps):
            frame = self.ctx.capture_bgr()
            det = self._tree_model.predict(frame, conf_threshold=0.5)
            if not det:
                self._walk(1.0)
                continue
            cx = det[0].center[0]
            offset = (cx - w / 2) / self.ctx.transform.scale  # → 1080p 像素语义
            if abs(offset) > 60:
                self.ctx.input.move_camera_by(offset * 0.8, 0)
                self.ctx.sleep(400)
            # 树框足够大说明已到跟前
            if det[0].width > 0.28 * w:
                return
            self._walk(1.2)

    # ---- 主流程 ----

    def _wait_for_reward_result_ready(
        self, cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Wait until the stable reward-page exit control is visible."""
        for _ in range(20):
            if cancelled and cancelled():
                return False
            if self.ctx.capture_region().find(REWARD_RESULT_EXIT).is_exist():
                return True
            self.ctx.sleep(300)
        return False

    def _recognize_rewards(self, cancelled: Callable[[], bool] | None = None) -> None:
        if not self.reward_recognition_enabled:
            return
        from .reward_result import RewardResultRecognizer

        if not self._wait_for_reward_result_ready(cancelled):
            self.log("[AutoDomain] 奖励结果页未检测到退出按钮，跳过本轮奖励识别")
            return
        try:
            if self._reward_recognizer is None:
                self._reward_recognizer = RewardResultRecognizer(self.ctx, log=self.log)
            rewards = self._reward_recognizer.recognize_multi_page(self.reward_max_pages)
        except Exception as error:
            self.log(f"[AutoDomain] 奖励识别失败，跳过本轮汇总：{error}")
            return
        for name, quantity in rewards.items():
            self.reward_summary[name] = self.reward_summary.get(name, 0) + quantity

    def run(self, cancelled: Callable[[], bool] | None = None) -> dict[str, int]:
        for rd in range(1, self.rounds + 1):
            if cancelled and cancelled():
                return dict(self.reward_summary)
            self.log(f"[AutoDomain] 第 {rd}/{self.rounds} 轮")

            # ① 启动挑战（面板按钮或场内交互）
            if not self._tap_text("启动挑战", "开始挑战", "单人挑战", timeout_s=8):
                self.ctx.input.key_press("F")  # 交互容错
            self.ctx.sleep(3000)
            self._tap_text("确认", roi=(1100, 600, 700, 400), timeout_s=4)
            self.ctx.sleep(4000)

            # ② 前进触发战斗
            self._walk(3.5)

            # ③ 战斗
            self.fight.run(cancelled=cancelled)

            # ④ 等「挑战达成」
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self._find_text("挑战达成", "挑战完成") is not None:
                    break
                self.ctx.sleep(1500)
            self.ctx.sleep(3000)

            # ⑤ 走向石化古树并交互（YOLO 检测古树转向，模型缺失时直走回退）
            self._walk_to_tree()
            self.ctx.input.key_press("F")
            self.ctx.sleep(2000)

            # ⑥ 树脂领奖
            resin = "使用浓缩树脂" if self.use_condensed else "使用原粹树脂"
            if not self._tap_text(resin, "浓缩树脂" if self.use_condensed else "原粹树脂",
                                  timeout_s=8):
                self.log("[AutoDomain] 未找到树脂选项（树脂不足或未到古树），本轮跳过领奖")
            self.ctx.sleep(3000)

            # 奖励页动画结束且卡片稳定后再识别，避免截到淡入中的图标。
            self._recognize_rewards(cancelled)

            # ⑦ 继续或退出
            if rd < self.rounds:
                if not self._tap_text("继续挑战", timeout_s=8):
                    self.log("[AutoDomain] 未能继续挑战，提前退出")
                    self._tap_text("退出秘境", timeout_s=5)
                    return dict(self.reward_summary)
                self.ctx.sleep(6000)
            else:
                self._tap_text("退出秘境", timeout_s=8)
                self.ctx.sleep(8000)
        self.log("[AutoDomain] 完成")
        return dict(self.reward_summary)
