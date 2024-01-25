from __future__ import annotations

from typing import Any

from speech2text.utils.sample_types import (
    CONVERT_FROM_INT32,
    CONVERT_FROM_TO,
    CONVERT_TO_INT32,
    TYPE_EXTEND_FUNC,
    TYPE_FACTORY,
)
from speech2text.utils.sample_types import SampleDType as Sdt


class Samples:
    _main_types = (
        Sdt.BYTES_2,
        Sdt.NP_I32,
        Sdt.NP_F32,
    )

    def __init__(self, init_data: Any = None, init_dtype: Sdt = None) -> None:
        if isinstance(init_data, Samples):
            self._sample_data = init_data._sample_data
            return

        self._sample_data = {}
        for st in self._main_types:
            self._sample_data[st] = TYPE_FACTORY[st]([])
        if not init_data:
            return
        if not init_dtype:
            init_dtype = Sdt.guess(init_data)
        self.update(init_data, init_dtype)

    def as_type(self, dtype: Sdt):
        if dtype not in self._sample_data:
            i32_data = self._sample_data[Sdt.NP_I32]
            dt_data = CONVERT_FROM_INT32[dtype](i32_data)
            self._sample_data[dtype] = dt_data
        return self._sample_data[dtype]

    def _drop_non_main_types(self):
        for st in self._sample_data:
            if st not in self._main_types:
                del self._sample_data[st]

    def update(self, data: Any, dtype: Sdt):
        self._sample_data[dtype] = data
        i32_data = CONVERT_TO_INT32[dtype](data)
        for st in self._sample_data:
            if st != dtype:
                st_data = CONVERT_FROM_INT32[st](i32_data)
                self._sample_data[st] = st_data

    def extend(self, data: Samples | Any, dtype: Sdt = None):
        if not dtype:
            if not isinstance(data, Samples):
                raise ValueError
            dtype = Sdt.NP_I32
            data = data._sample_data[dtype]

        TYPE_EXTEND_FUNC[dtype](self._sample_data[dtype], data)
        i32_data = CONVERT_TO_INT32[dtype](data)
        for st in self._sample_data:
            if st != dtype:
                st_data = CONVERT_FROM_INT32[st](i32_data)
                TYPE_EXTEND_FUNC[st](self._sample_data[st], st_data)

    def __len__(self):
        return len(self._sample_data[Sdt.NP_I32])
