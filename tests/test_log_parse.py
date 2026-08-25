import json
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace


def _write_log(path: Path) -> Path:
    path.write_text(
        "\n".join([
            '[08:00:00.100] [info] 配置组 "每日锄地" 加载完成，共2个脚本',
            '[08:00:01.000] [task] → 开始执行地图追踪任务: "路线一.json"',
            '[08:00:02.000] [task] 交互或拾取："优兰尼娅湖"',
            '[08:00:03.000] [task] 交互或拾取："优兰尼娅湖"',
            '[08:00:04.000] [task] 传送失败，重试 2 次',
            '[08:00:05.000] [task] 疑似卡死，尝试脱离...',
            '[08:00:06.000] [task] 执行脚本时发生异常: "示例错误"',
            '[08:00:07.000] [task] → 脚本执行结束: "路线一.json"',
            '[08:00:08.000] [task] → 开始执行 JS脚本: "奖励.js"',
            '[08:00:10.000] [task] → 脚本执行结束: "奖励.js"',
            '[08:00:11.000] [info] 配置组 "每日锄地" 执行结束',
        ]) + "\n",
        encoding="utf-8",
    )
    return path


def test_log_parser_matches_group_task_picks_and_faults(tmp_path: Path):
    from bgi_touch.tasks.log_parse import parse_log_files

    path = _write_log(tmp_path / "better-genshin-impact20260825.log")
    report = parse_log_files([path])

    assert len(report.groups) == 1
    group = report.groups[0]
    assert group.name == "每日锄地"
    assert group.declared_script_count == 2
    assert group.duration_seconds == 10.9
    assert [task.name for task in group.tasks] == ["路线一.json", "奖励.js"]
    first = group.tasks[0]
    assert first.duration_seconds == 6
    assert first.picks == {"优兰尼娅湖": 2}
    assert first.fault.pathing_success_end is True
    assert first.fault.teleport_fail_count == 2
    assert first.fault.stuck_count == 1
    assert first.fault.error_count == 1
    assert report.pick_totals == {"优兰尼娅湖": 2}
    assert report.fault_totals["errCount"] == 1


def test_log_parser_keeps_instances_independent(tmp_path: Path):
    from bgi_touch.tasks.log_parse import parse_log_files

    path = tmp_path / "better-genshin-impact20260825_001.log"
    path.write_text(
        "\n".join([
            '[09:00:00] [info] [Main:S1:P1:T1] 配置组 "甲" 加载完成，共1个脚本',
            '[09:00:01] [task] [Main:S1:P1:T1] → 开始执行地图追踪任务: "甲.json"',
            '[09:00:02] [task] [Main:S1:P1:T1] → 脚本执行结束: "甲.json"',
            '[09:00:03] [info] [Main:S1:P1:T1] 配置组 "甲" 执行结束',
            '[09:00:00] [info] [Main:S2:P1:T1] 配置组 "乙" 加载完成，共1个脚本',
            '[09:00:01] [task] [Main:S2:P1:T1] → 开始执行地图追踪任务: "乙.json"',
            '[09:00:04] [task] [Main:S2:P1:T1] 此追踪脚本未正常走完！',
            '[09:00:05] [task] [Main:S2:P1:T1] → 脚本执行结束: "乙.json"',
            '[09:00:06] [info] [Main:S2:P1:T1] 配置组 "乙" 执行结束',
        ]) + "\n",
        encoding="utf-8",
    )

    report = parse_log_files([path])
    assert [group.name for group in report.groups] == ["甲", "乙"]
    assert report.groups[1].tasks[0].fault.pathing_success_end is False


def test_log_parser_closes_incomplete_group_at_last_timestamp(tmp_path: Path):
    from bgi_touch.tasks.log_parse import parse_log_files

    path = tmp_path / "manual-export.log"
    path.write_text(
        '[10:00:00] [info] 配置组 "中断" 加载完成，共1个脚本\n'
        '[10:00:02] [task] → 开始执行 JS脚本: "main.js"\n',
        encoding="utf-8",
    )
    report = parse_log_files([(path, date(2026, 8, 25))])
    group = report.groups[0]
    assert group.end_date == group.tasks[0].end_date
    assert group.end_date.isoformat() == "2026-08-25T10:00:02"


def test_log_discovery_and_duration_formatting(tmp_path: Path):
    from bgi_touch.tasks.log_parse import discover_log_files, format_duration

    older = tmp_path / "better-genshin-impact20260824.log"
    newer = tmp_path / "better-genshin-impact20260825_002.log"
    older.write_text("", encoding="utf-8")
    newer.write_text("", encoding="utf-8")
    (tmp_path / "other.log").write_text("", encoding="utf-8")

    discovered = discover_log_files(tmp_path)
    assert [item[0].name for item in discovered] == [older.name, newer.name]
    assert format_duration(3661.5) == "1小时1分钟1.50秒"
    assert format_duration(0) == "0秒"


def test_log_parse_cli_is_device_free(tmp_path: Path, capsys):
    from bgi_touch.cli import cmd_log_parse

    path = _write_log(tmp_path / "better-genshin-impact20260825.log")
    args = SimpleNamespace(sources=[str(path)], date=None, format="json")
    assert cmd_log_parse(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["groupCount"] == 1


def test_log_report_html_is_self_contained_and_escapes_log_content():
    from bgi_touch.tasks.log_parse import ConfigGroupLog, ConfigTaskLog, LogParseReport

    task = ConfigTaskLog('<任务 "A">')
    task.add_pick('<img src=x onerror=alert(1)>')
    task.fault.error_count = 1
    report = LogParseReport(
        groups=[ConfigGroupLog('<配置组>', tasks=[task])],
        sources=['/tmp/<日志>.log'],
    )
    html = report.to_html(title='<报告>')

    assert html.startswith("<!doctype html>")
    assert "<title>&lt;报告&gt;</title>" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<img src=x" not in html
    assert "<script>" in html
    assert "https://" not in html
    assert "sortTable" in html
    assert "远程" not in html
    assert ">异常</th>" in html
    assert ">异常</th>" not in report.to_html(include_faults=False)


def test_log_parse_html_cli_is_device_free(tmp_path: Path, capsys):
    from bgi_touch.cli import cmd_log_parse

    path = _write_log(tmp_path / "better-genshin-impact20260825.log")
    args = SimpleNamespace(
        sources=[str(path)], date=None, format="html", title="离线报告", no_faults=False,
    )
    assert cmd_log_parse(args) == 0
    output = capsys.readouterr().out
    assert "<!doctype html>" in output
    assert "<title>离线报告</title>" in output
    assert "配置组：每日锄地" in output


def test_log_report_includes_sortable_mora_statistics_by_custom_day():
    from bgi_touch.tasks.log_parse import ConfigGroupLog, ConfigTaskLog, LogParseReport
    from bgi_touch.tasks.travel_diary import ActionItem

    task = ConfigTaskLog(
        "路线.json",
        start_date=datetime(2026, 8, 25, 4, 30),
        end_date=datetime(2026, 8, 25, 6, 0),
    )
    group = ConfigGroupLog(
        "每日锄地",
        start_date=datetime(2026, 8, 25, 4, 0),
        end_date=datetime(2026, 8, 25, 6, 30),
        tasks=[task],
    )
    report = LogParseReport(
        groups=[group],
        mora_items=(
            ActionItem(37, "小怪", "2026-08-25 03:30:00", 100),
            ActionItem(37, "精英", "2026-08-25 05:00:00", 200),
            ActionItem(28, "突发", "2026-08-25 05:10:00", 15),
        ),
    )

    assert [item.day.isoformat() for item in report.mora_day_statistics] == [
        "2026-08-24", "2026-08-25",
    ]
    assert report.mora_statistics.total_mora_killing_monsters == 300
    payload = report.to_dict()
    assert payload["mora"]["totals"]["totalMora"] == 300
    html = report.to_html()
    assert 'data-sort-type="date"' in html
    assert 'data-sort-type="number"' in html
    assert "按日摩拉收益统计" in html
    assert "锄地总计：小怪" in html
    assert "摩拉（每秒）" in html


def test_log_parse_loads_local_diary_cache_for_html(tmp_path: Path, capsys):
    from bgi_touch.cli import cmd_log_parse

    log_path = _write_log(tmp_path / "better-genshin-impact20260825.log")
    diary_path = tmp_path / "2026_08.json"
    diary_path.write_text(json.dumps({
        "data": {"list": [
            {"action_id": 37, "action": "小怪", "time": "2026-08-25 08:00:02", "num": 100},
        ]},
    }, ensure_ascii=False), encoding="utf-8")
    args = SimpleNamespace(
        sources=[str(log_path)], date=None, format="html", title="带札记", no_faults=False,
        diary_file=[str(diary_path)], diary_cache_dir=None, game_uid=None,
    )

    assert cmd_log_parse(args) == 0
    output = capsys.readouterr().out
    assert "按日摩拉收益统计" in output
    assert "带札记" in output
