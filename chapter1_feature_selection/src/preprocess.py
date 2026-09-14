# src/preprocess.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureMeta:
    feature_dim: int
    field_slices: Dict[str, Tuple[int, int]]  # field -> [start, end)
    fields_order: List[str]                   # 保持稳定顺序


def _safe_get(d: Dict[str, Any], key: str) -> Any:
    return d.get(key, None)


def _agg_numeric(values: List[float]) -> np.ndarray:
    """输出4维统计: mean/std/max/min"""
    if len(values) == 0:
        return np.zeros(4, dtype=np.float32)
    arr = np.array(values, dtype=np.float32)
    return np.array([arr.mean(), arr.std(), arr.max(), arr.min()], dtype=np.float32)


class WindowFeaturizer:
    """
    把一个窗口(events列表)转成统一向量：
    - numeric_fields: 4维统计
    - categorical_fields: 每字段一个 hasher 输出 hash_dim_cat
    - text_fields: 每字段一个 hasher 输出 hash_dim_text（把文本按token做hash）
    """
    def __init__(
        self,
        numeric_fields: List[str],
        categorical_fields: List[str],
        text_fields: List[str],
        hash_dim_cat: int = 256,
        hash_dim_text: int = 256,
    ):
        self.numeric_fields = numeric_fields
        self.categorical_fields = categorical_fields
        self.text_fields = text_fields

        self.hash_dim_cat = hash_dim_cat
        self.hash_dim_text = hash_dim_text

        self.cat_hashers = {f: FeatureHasher(n_features=hash_dim_cat, input_type="string") for f in categorical_fields}
        self.text_hashers = {f: FeatureHasher(n_features=hash_dim_text, input_type="string") for f in text_fields}

        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self._fitted = False

        # 生成字段切片（保证字段级mask可用）
        self.meta = self._build_meta()

    def _build_meta(self) -> FeatureMeta:
        field_slices: Dict[str, Tuple[int, int]] = {}
        fields_order: List[str] = []
        cursor = 0

        # numeric: 每字段4维
        for f in self.numeric_fields:
            fields_order.append(f)
            field_slices[f] = (cursor, cursor + 4)
            cursor += 4

        # categorical: 每字段 hash_dim_cat
        for f in self.categorical_fields:
            fields_order.append(f)
            field_slices[f] = (cursor, cursor + self.hash_dim_cat)
            cursor += self.hash_dim_cat

        # text: 每字段 hash_dim_text
        for f in self.text_fields:
            fields_order.append(f)
            field_slices[f] = (cursor, cursor + self.hash_dim_text)
            cursor += self.hash_dim_text

        return FeatureMeta(feature_dim=cursor, field_slices=field_slices, fields_order=fields_order)

    def featurize_one(self, events: List[Dict[str, Any]]) -> np.ndarray:
        x = np.zeros(self.meta.feature_dim, dtype=np.float32)

        # 1) numeric
        for f in self.numeric_fields:
            vals = []
            for e in events:
                v = _safe_get(e, f)
                if isinstance(v, (int, float)) and not np.isnan(v):
                    vals.append(float(v))
            start, end = self.meta.field_slices[f]
            x[start:end] = _agg_numeric(vals)

        # 2) categorical (每个字段统计出现次数，hash到向量)
        for f in self.categorical_fields:
            tokens = []
            for e in events:
                v = _safe_get(e, f)
                if v is None:
                    continue
                tokens.append(str(v))
            start, end = self.meta.field_slices[f]
            if len(tokens) > 0:
                vec = self.cat_hashers[f].transform([tokens]).toarray().astype(np.float32)[0]
                x[start:end] = vec

        # 3) text（简单按空格切词，hash）
        for f in self.text_fields:
            tokens = []
            for e in events:
                v = _safe_get(e, f)
                if v is None:
                    continue
                text = str(v)
                # 简单token化：先跑通，后面你可替换更强的分词/解析
                tokens.extend([t for t in text.replace("/", " ").replace("=", " ").split() if t])
            start, end = self.meta.field_slices[f]
            if len(tokens) > 0:
                vec = self.text_hashers[f].transform([tokens]).toarray().astype(np.float32)[0]
                x[start:end] = vec

        return x

    def fit_transform(self, windows_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, FeatureMeta]:
        """
        输入：windows.parquet的DataFrame（含events,json字符串）
        输出：X,y,meta
        """
        X_list = []
        y_list = []

        for _, row in windows_df.iterrows():
            events = json.loads(row["events"])
            x = self.featurize_one(events)
            X_list.append(x)
            y_list.append(int(row["y"]))

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

        # numeric部分做标准化（hash部分不建议做标准化）
        if len(self.numeric_fields) > 0:
            num_dim = 4 * len(self.numeric_fields)
            X[:, :num_dim] = self.scaler.fit_transform(X[:, :num_dim])
            self._fitted = True

        return X, y, self.meta
