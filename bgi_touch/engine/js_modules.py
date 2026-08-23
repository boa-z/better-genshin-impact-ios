"""Small, sandboxed ES-module bridge for PythonMonkey.

PythonMonkey 1.x embeds SpiderMonkey as a script engine and exposes CommonJS,
but it does not parse static ``import``/``export`` declarations. BetterGI
community scripts use a deliberately small ES-module subset, so we translate
that subset to synchronous module wrappers while preserving exported bindings
through getters.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator

from .recognition import Mat


_IMPORT_RE = re.compile(
    r"^[ \t]*import[ \t]+"
    r"(?:(?P<clause>.*?)[ \t\r\n]+from[ \t\r\n]+)?"
    r"(?P<quote>['\"])(?P<spec>[^'\"]+)(?P=quote)[ \t]*;?[ \t]*(?:\r?\n|$)",
    re.MULTILINE | re.DOTALL,
)


def import_specifiers(source: str) -> Iterator[str]:
    for match in _IMPORT_RE.finditer(source):
        yield match.group("spec")


def rewrite_import_specifiers(source: str, rewrite: Callable[[str], str]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = rewrite(match.group("spec"))
        start, end = match.span("spec")
        whole_start = match.start()
        relative_start, relative_end = start - whole_start, end - whole_start
        text = match.group(0)
        return text[:relative_start] + value + text[relative_end:]

    return _IMPORT_RE.sub(replace, source)


def _named_imports(body: str) -> str:
    values = []
    for raw in body.split(","):
        item = raw.strip()
        if not item:
            continue
        parts = re.split(r"\s+as\s+", item, maxsplit=1)
        values.append(parts[0] if len(parts) == 1 else f"{parts[0]}: {parts[1]}")
    return ", ".join(values)


def _import_statement(clause: str | None, specifier: str, index: int) -> str:
    encoded = json.dumps(specifier, ensure_ascii=False)
    if not clause:
        return f"__bgi_require({encoded});"
    clause = " ".join(clause.split())
    module_name = f"__bgi_module_{index}"
    lines = [f"const {module_name} = __bgi_require({encoded});"]
    if clause.startswith("{"):
        lines.append(f"const {{{_named_imports(clause[1:-1])}}} = {module_name};")
    elif clause.startswith("* as "):
        lines.append(f"const {clause[5:].strip()} = {module_name};")
    elif "," in clause:
        default_name, rest = clause.split(",", 1)
        lines.append(
            f"const {default_name.strip()} = Object.prototype.hasOwnProperty.call("
            f"{module_name}, 'default') ? {module_name}.default : {module_name};"
        )
        rest = rest.strip()
        if rest.startswith("{"):
            lines.append(f"const {{{_named_imports(rest[1:-1])}}} = {module_name};")
        elif rest.startswith("* as "):
            lines.append(f"const {rest[5:].strip()} = {module_name};")
    else:
        lines.append(
            f"const {clause} = Object.prototype.hasOwnProperty.call("
            f"{module_name}, 'default') ? {module_name}.default : {module_name};"
        )
    return "\n".join(lines)


def extract_imports(source: str) -> tuple[str, str]:
    statements: list[str] = []

    def replace(match: re.Match[str]) -> str:
        statements.append(_import_statement(
            match.group("clause"), match.group("spec"), len(statements),
        ))
        # Keep line numbers useful in JavaScript stack traces.
        return "\n" * match.group(0).count("\n")

    body = _IMPORT_RE.sub(replace, source)
    return "\n".join(statements), body


def transform_exports(source: str) -> str:
    exports: list[tuple[str, str]] = []

    def declaration(match: re.Match[str]) -> str:
        exports.append((match.group("name"), match.group("name")))
        return f"{match.group('indent')}{match.group('declaration')}"

    source = re.sub(
        r"^(?P<indent>[ \t]*)export[ \t]+"
        r"(?P<declaration>(?:(?:async[ \t]+)?function|class)[ \t]+"
        r"(?P<name>[A-Za-z_$][\w$]*))",
        declaration,
        source,
        flags=re.MULTILINE,
    )
    source = re.sub(
        r"^(?P<indent>[ \t]*)export[ \t]+"
        r"(?P<declaration>(?:const|let|var)[ \t]+(?P<name>[A-Za-z_$][\w$]*))",
        declaration,
        source,
        flags=re.MULTILINE,
    )

    default_name = "__bgi_default_export"
    source, default_count = re.subn(
        r"^[ \t]*export[ \t]+default[ \t]+",
        f"const {default_name} = ",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if default_count:
        exports.append(("default", default_name))

    def export_list(match: re.Match[str]) -> str:
        for raw in match.group("body").split(","):
            item = raw.strip()
            if not item:
                continue
            parts = re.split(r"\s+as\s+", item, maxsplit=1)
            local = parts[0].strip()
            exported = parts[-1].strip()
            exports.append((exported, local))
        return "\n" * match.group(0).count("\n")

    source = re.sub(
        r"^[ \t]*export[ \t]*\{(?P<body>.*?)\}[ \t]*;?[ \t]*(?:\r?\n|$)",
        export_list,
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    seen = set()
    bindings = []
    for exported, local in exports:
        if exported in seen:
            continue
        seen.add(exported)
        bindings.append(
            "Object.defineProperty(exports, "
            f"{json.dumps(exported)}, {{enumerable:true,get:()=>{local}}});"
        )
    return source + "\n" + "\n".join(bindings)


class JsModuleLoader:
    def __init__(self, pm, script_dir: str | Path, manifest: dict,
                 wrap: Callable[[Any], Any]):
        self.pm = pm
        self.script_dir = Path(script_dir).resolve()
        self.wrap = wrap
        self.cache: dict[Path, Any] = {}
        self.roots = [self.script_dir]
        for value in manifest.get("library", []) or []:
            root = (self.script_dir / str(value)).resolve()
            if root.is_relative_to(self.script_dir) and root.is_dir():
                self.roots.append(root)
        vendor = self.script_dir / ".bgi-touch-vendor"
        if vendor.is_dir():
            self.roots.append(vendor.resolve())
        # bettergi-scripts-list keeps trusted shared modules in a repository
        # level ``packages`` directory referenced by ../../../packages/...
        for ancestor in self.script_dir.parents:
            packages = ancestor / "packages"
            if packages.is_dir() and (ancestor / "repo" / "js").is_dir():
                self.roots.append(packages.resolve())
                break

    def _allowed(self, path: Path) -> bool:
        return any(path.is_relative_to(root) for root in self.roots)

    @staticmethod
    def _candidates(path: Path) -> list[Path]:
        values = [path]
        if not path.suffix:
            values.extend((path.with_suffix(".js"), path.with_suffix(".json"), path / "index.js"))
        return values

    def resolve(self, specifier: str, importer: Path) -> Path:
        value = str(specifier)
        if value.startswith(("node:", "http://", "https://")):
            raise PermissionError(f"JS 模块不允许外部协议: {value}")
        bases = [importer.parent] if value.startswith(("./", "../")) else self.roots
        for base in bases:
            for candidate in self._candidates((base / value).resolve()):
                if candidate.is_file() and self._allowed(candidate):
                    return candidate
        raise FileNotFoundError(f"无法解析 JS 模块 {value}（来源 {importer.name}）")

    def require(self, specifier: str, importer: Path) -> Any:
        path = self.resolve(specifier, importer)
        if path in self.cache:
            return self.cache[path]
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            module = self.pm.eval("value => ({default:value})")(self.wrap(Mat.from_file(str(path))))
            self.cache[path] = module
            return module
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            module = self.pm.eval("value => ({default:value})")(value)
            self.cache[path] = module
            return module
        if path.suffix.lower() != ".js":
            raise ValueError(f"不支持的 JS 模块资源类型: {path.suffix}")

        exports = self.pm.eval("({})")
        module = self.pm.eval("exports => ({exports})")(exports)
        self.cache[path] = exports
        prelude, body = extract_imports(path.read_text(encoding="utf-8-sig"))
        code = transform_exports(prelude + "\n" + body)
        wrapper = self.pm.eval(
            "(function(exports,module,__bgi_require,__filename,__dirname){\n"
            "'use strict';\n" + code + "\nreturn module.exports;\n})",
            {"filename": str(path)},
        )
        local_require = lambda value: self.require(str(value), path)
        try:
            result = wrapper(exports, module, local_require, str(path), str(path.parent))
            self.cache[path] = result
            return result
        except Exception:
            self.cache.pop(path, None)
            raise
