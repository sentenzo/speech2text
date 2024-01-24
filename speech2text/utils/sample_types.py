from enum import Enum, auto
from typing import Any

import numpy as np

MAX_INT32 = 1 << 31


class SampleDType(Enum):
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
            return SampleDType.BYTES_2
        if not isinstance(data, np.ndarray | list):
            raise ValueError
        val = data[0]
        if isinstance(val, np.int16 | int):
            return SampleDType.NP_I16
        if isinstance(val, np.int32):
            return SampleDType.NP_I32
        if isinstance(val, np.float16):
            return SampleDType.NP_F16
        if isinstance(val, np.float32 | float):
            return SampleDType.NP_F32
        raise ValueError


def _b4_to_i32(data: bytes | bytearray) -> np.int32:
    return np.frombuffer(data, "<i4")


def _i16_to_i32(data: np.int16) -> np.int32:
    return np.array(data).astype(np.int32) << 16


def _b2_to_i32(data: bytes | bytearray) -> np.int32:
    data = np.frombuffer(data, "<i2")  # little endian
    return _i16_to_i32(data)


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
    SampleDType.BYTES_2: _b2_to_i32,
    SampleDType.BYTES_4: _b4_to_i32,
    SampleDType.NP_I16: _i16_to_i32,
    SampleDType.NP_I32: _i32_to_i32,
    SampleDType.NP_F16: _f16_to_i32,
    SampleDType.NP_F32: _f32_to_i32,
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
    SampleDType.BYTES_2: _i32_to_b2,
    SampleDType.BYTES_4: _i32_to_b4,
    SampleDType.NP_I16: _i32_to_i16,
    SampleDType.NP_I32: _i32_to_i32,
    SampleDType.NP_F16: _i32_to_f16,
    SampleDType.NP_F32: _i32_to_f32,
}

CONVERT_FROM_TO = {}
for frm in SampleDType:
    CONVERT_FROM = CONVERT_FROM_TO.get(frm, {})
    for to in SampleDType:

        def convert(data):
            data = CONVERT_TO_INT32[frm](data)
            return CONVERT_FROM_INT32[to](data)

        CONVERT_FROM[to] = convert

TYPE_FACTORY = {
    SampleDType.BYTES_2: bytearray,
    SampleDType.BYTES_4: bytearray,
    SampleDType.NP_I16: lambda arr: np.array(arr, "<i2"),
    SampleDType.NP_I32: lambda arr: np.array(arr, "<i4"),
    SampleDType.NP_F16: lambda arr: np.array(arr, "f2"),
    SampleDType.NP_F32: lambda arr: np.array(arr, "f4"),
}

BYTES_EXTEND = lambda arr, add: arr.extend(add)
NP_EXTEND = lambda arr, add: np.append(arr, add)

TYPE_EXTEND_FUNC = {
    SampleDType.BYTES_2: BYTES_EXTEND,
    SampleDType.BYTES_4: BYTES_EXTEND,
    SampleDType.NP_I16: NP_EXTEND,
    SampleDType.NP_I32: NP_EXTEND,
    SampleDType.NP_F16: NP_EXTEND,
    SampleDType.NP_F32: NP_EXTEND,
}
