from __future__ import annotations

from enum import Enum, auto
from typing import Any

import numpy as np

MAX_INT32 = 1 << 31


class SampleDataType(Enum):
    BYTES_2 = auto()
    BYTES_4 = auto()
    NP_I16 = np.int16
    NP_I32 = np.int32
    NP_F16 = np.float16
    NP_F32 = np.float32

    @staticmethod
    def guess(data: Any):
        if not data:
            raise ValueError
        if isinstance(data, bytes | bytearray):
            return SampleDataType.BYTES
        if not isinstance(data, np.ndarray | list):
            raise ValueError
        val = data[0]
        if isinstance(val, np.int16 | int):
            return SampleDataType.NP_I16
        if isinstance(val, np.int32):
            return SampleDataType.NP_I32
        if isinstance(val, np.float16):
            return SampleDataType.NP_F16
        if isinstance(val, np.float32 | float):
            return SampleDataType.NP_F32


def _b2_to_i32(data: bytes | bytearray) -> np.int32:
    np_datatype = np.dtype(np.int16)
    np_datatype = np_datatype.newbyteorder("L")  # little endian
    data = np.frombuffer(data, np_datatype)
    return data.astype(np.int32)


def _b4_to_i32(data: bytes | bytearray) -> np.int32:
    np_datatype = np.dtype(np.int32)
    np_datatype = np_datatype.newbyteorder("L")  # little endian
    data = np.frombuffer(data, np_datatype)
    return data


def _i16_to_i32(data: np.int16) -> np.int32:
    return np.array(data).astype(np.int32)


def _i32_to_i32(data: np.int32 | int) -> np.int32:
    # return data
    return np.array(data)


def _f16_to_i32(data: np.float16) -> np.int32:
    data = np.array(data).astype(np.float32)
    data *= MAX_INT32
    return data.astype(np.int32)


def _f32_to_i32(data: np.float32 | float) -> np.int32:
    data = np.array(data) * MAX_INT32
    return data.astype(np.int32)


CONVERT_TO_INT32 = {
    SampleDataType.BYTES_2: _b2_to_i32,
    SampleDataType.BYTES_4: _b4_to_i32,
    SampleDataType.NP_I16: _i16_to_i32,
    SampleDataType.NP_I32: _i32_to_i32,
    SampleDataType.NP_F16: _f16_to_i32,
    SampleDataType.NP_F32: _f32_to_i32,
}


def _i32_to_b2(data: np.int32 | int) -> bytes:
    data = np.array(data, dtype="<i4")  # little endian int32
    return data.tobytes()


def _i32_to_i16(data: np.int32 | int) -> np.int16:
    data = np.array(data, dtype="<i4")
    data >>= 16
    return data.astype("<i2")


def _i32_to_b4(data: np.int32 | int) -> bytes:
    return _i32_to_i16(data).tobytes()


def _i32_to_f16(data: np.int32 | int) -> np.float16:
    data = np.array(data, dtype="<i4")
    return np.array(data / MAX_INT32, dtype="f2")


def _i32_to_f32(data: np.int32 | int) -> np.float32:
    data = np.array(data, dtype="<i4")
    return np.array(data / MAX_INT32, dtype="f4")


CONVERT_FROM_INT32 = {
    SampleDataType.BYTES_2: _i32_to_b2,
    SampleDataType.BYTES_4: _i32_to_b4,
    SampleDataType.NP_I16: _i32_to_i16,
    SampleDataType.NP_I32: _i32_to_i32,
    SampleDataType.NP_F16: _i32_to_f16,
    SampleDataType.NP_F32: _i32_to_f32,
}
