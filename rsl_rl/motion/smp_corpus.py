from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SMPDatasetEntry:
    """描述一个待导出的或待训练的 SMP 数据片段。"""

    name: str
    path: Path
    style_name: str
    style_id: int | None = None
    weight: float = 1.0

    def to_manifest_dict(self, root_dir: Path | None = None) -> dict[str, Any]:
        """将条目转成可写入 JSON 的字典。"""
        path_value = str(self.path)
        if root_dir is not None:
            try:
                path_value = str(self.path.relative_to(root_dir))
            except ValueError:
                path_value = str(self.path)
        return {
            "name": self.name,
            "path": path_value,
            "style": self.style_name,
            "style_id": self.style_id,
            "weight": self.weight,
        }


class SMPMotionCorpus:
    """多风格 SMP 语料描述。

    负责两件事：
    1. 解析 manifest，把相对路径解析为绝对路径；
    2. 为风格名建立稳定的 style_id 映射，并允许同风格绑定多个 shard。
    """

    def __init__(self, entries: list[SMPDatasetEntry]):
        if not entries:
            raise ValueError("SMPMotionCorpus requires at least one dataset entry")

        explicit_style_to_id: dict[str, int] = {}
        for entry in entries:
            if entry.style_id is None:
                continue
            previous = explicit_style_to_id.get(entry.style_name)
            if previous is not None and previous != entry.style_id:
                raise ValueError(
                    f"Style '{entry.style_name}' has inconsistent ids: {previous} vs {entry.style_id}"
                )
            explicit_style_to_id[entry.style_name] = entry.style_id

        style_names = sorted({entry.style_name for entry in entries})
        if explicit_style_to_id and len(explicit_style_to_id) == len(style_names):
            self.style_to_id = dict(explicit_style_to_id)
        else:
            self.style_to_id = {name: idx for idx, name in enumerate(style_names)}

        self.entries = [replace(entry, style_id=self.style_to_id[entry.style_name]) for entry in entries]
        self.style_names = sorted(self.style_to_id, key=self.style_to_id.get)
        self.entries_by_style = {
            style_name: [entry for entry in self.entries if entry.style_name == style_name]
            for style_name in self.style_names
        }

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "SMPMotionCorpus":
        """从 JSON manifest 构建语料对象。"""
        manifest_path = Path(manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = payload.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("Manifest must contain a non-empty 'datasets' list")

        entries: list[SMPDatasetEntry] = []
        for raw_entry in datasets:
            style_name = raw_entry.get("style") or raw_entry.get("style_name")
            if not style_name:
                raise ValueError(f"Manifest entry is missing style: {raw_entry}")
            entry_path = Path(raw_entry["path"])
            if not entry_path.is_absolute():
                entry_path = (manifest_path.parent / entry_path).resolve()
            entries.append(
                SMPDatasetEntry(
                    name=raw_entry.get("name", entry_path.stem),
                    path=entry_path,
                    style_name=str(style_name),
                    style_id=raw_entry.get("style_id"),
                    weight=float(raw_entry.get("weight", 1.0)),
                )
            )
        return cls(entries)
