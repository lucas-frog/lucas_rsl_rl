from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rsl_rl.motion.smp_corpus import SMPMotionCorpus


@dataclass(frozen=True)
class _SMPClipRecord:
    """保存单个 clip 的帧数据和静态元数据。"""

    frames: torch.Tensor
    style_id: int | None
    style_name: str | None
    source_name: str
    clip_id: int


@dataclass(frozen=True)
class _SMPWindowRecord:
    """保存窗口索引，避免重复切片和重复拷贝元数据。"""

    clip_index: int
    start: int
    stop: int


class SMPMotionWindowDataset(Dataset):
    """从单文件或多风格 manifest 中构建 SMP 滑动窗口数据集。"""

    def __init__(self, path: str | Path, window_size: int = 10, stride: int = 1):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        dataset_path = Path(path)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"SMP dataset file not found: {dataset_path}")

        self.window_size = window_size
        self.stride = stride
        self.clips: list[_SMPClipRecord] = []
        self.window_records: list[_SMPWindowRecord] = []
        self.style_names: list[str] = []
        self.style_to_id: dict[str, int] = {}

        if dataset_path.suffix.lower() == ".json":
            self._load_from_manifest(dataset_path)
        else:
            frames, metadata = self._load_npz_frames(dataset_path)
            self.style_names = [metadata["style_name"]] if metadata["style_name"] is not None else []
            self.style_to_id = (
                {metadata["style_name"]: metadata["style_id"]}
                if metadata["style_name"] is not None and metadata["style_id"] is not None
                else {}
            )
            self._append_clip(
                frames=frames,
                style_id=metadata["style_id"],
                style_name=metadata["style_name"],
                source_name=metadata["source_name"] or dataset_path.stem,
                clip_id=0,
            )

    def _load_from_manifest(self, manifest_path: Path) -> None:
        corpus = SMPMotionCorpus.from_manifest(manifest_path)
        self.style_names = list(corpus.style_names)
        self.style_to_id = dict(corpus.style_to_id)
        for clip_id, entry in enumerate(corpus.entries):
            frames, metadata = self._load_npz_frames(entry.path)
            self._append_clip(
                frames=frames,
                style_id=entry.style_id,
                style_name=entry.style_name,
                source_name=metadata["source_name"] or entry.name,
                clip_id=clip_id,
            )

    @staticmethod
    def _read_optional_scalar(data: np.lib.npyio.NpzFile, key: str):
        if key not in data:
            return None
        value = np.asarray(data[key]).reshape(-1)[0]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "item"):
            return value.item()
        return value

    def _load_npz_frames(self, dataset_path: Path) -> tuple[torch.Tensor, dict[str, object]]:
        with np.load(dataset_path) as data:
            if "frames" not in data:
                raise KeyError("SMP dataset npz must contain 'frames'")
            frames = data["frames"]
            metadata = {
                "style_id": self._read_optional_scalar(data, "style_id"),
                "style_name": self._read_optional_scalar(data, "style_name"),
                "source_name": self._read_optional_scalar(data, "source_name"),
            }

        if frames.ndim != 2:
            raise ValueError(f"Expected frames to have shape (num_frames, feature_dim), got {frames.shape}")
        if frames.shape[0] < self.window_size:
            raise ValueError(f"Expected at least {self.window_size} frames, got {frames.shape[0]}")

        return torch.as_tensor(frames, dtype=torch.float32), metadata

    def _append_clip(
        self,
        frames: torch.Tensor,
        style_id: int | None,
        style_name: str | None,
        source_name: str,
        clip_id: int,
    ) -> None:
        clip_index = len(self.clips)
        self.clips.append(
            _SMPClipRecord(
                frames=frames,
                style_id=None if style_id is None else int(style_id),
                style_name=None if style_name is None else str(style_name),
                source_name=str(source_name),
                clip_id=clip_id,
            )
        )
        for start in range(0, frames.shape[0] - self.window_size + 1, self.stride):
            self.window_records.append(_SMPWindowRecord(clip_index=clip_index, start=start, stop=start + self.window_size))

    def __len__(self) -> int:
        return len(self.window_records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.window_records[index]
        clip = self.clips[record.clip_index]
        return {
            "motion": clip.frames[record.start : record.stop],
            "style_id": clip.style_id,
            "style_name": clip.style_name,
            "clip_id": clip.clip_id,
            "source_name": clip.source_name,
        }
