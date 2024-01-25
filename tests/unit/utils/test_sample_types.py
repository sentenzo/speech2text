import numpy as np
import pytest

import speech2text.utils.sample_types as st
from speech2text.utils.sample_types import SampleDType as Sdt


@pytest.mark.parametrize(
    "data,dtype",
    [
        (b"\x00\xff", Sdt.BYTES_2),
        (b"", Sdt.BYTES_2),
        ([1, 2, 3], Sdt.NP_I32),
        (np.array([1, 2, 3], "i4"), Sdt.NP_I32),
        (np.array([1, 2, 3], "i2"), Sdt.NP_I16),
        (np.array([0.1, 0.2, 0.3], "f2"), Sdt.NP_F16),
        (np.array([0.1, 0.2, 0.3], "f4"), Sdt.NP_F32),
    ],
)
def test_sdt_guess(data, dtype):
    assert dtype == Sdt.guess(data)


TEST_SAMPLE_F = [-1.0, -0.5, -0.05, 0.0, 0.05, 0.5, 1.0]

POW_2_15 = 1 << 15
TEST_SAMPLE_I16 = [
    -POW_2_15,
    -POW_2_15 // 2,
    -POW_2_15 // 20,
    0,
    POW_2_15 // 20,
    POW_2_15 // 2,
    POW_2_15 - 1,
]

POW_2_31 = 1 << 31
TEST_SAMPLE_I32 = [
    -POW_2_31,
    -POW_2_31 // 2,
    -POW_2_31 // 20,
    0,
    POW_2_31 // 20,
    POW_2_31 // 2,
    POW_2_31 - 1,
]

TEST_SAMPLE_B2 = [
    b"\x00\x80",  # -32768
    b"\x00\xc0",  # -16384
    b"\x99\xf9",  # - 1639
    b"\x00\x00",  #      0
    b"\x66\x06",  # + 1638
    b"\x00\x40",  # +16384
    b"\xff\x7f",  # +32767
]
TEST_SAMPLE_B2 = b"".join(TEST_SAMPLE_B2)

TEST_SAMPLE_B4 = [
    b"\x00\x00\x00\x80",  # -2147483648 (-(1<<31))
    b"\x00\x00\x00\xc0",  # -1073741824 (-(1<<31)//2)
    b"\x99\x99\x99\xf9",  # - 107374183 (-(1<<31)//20)
    b"\x00\x00\x00\x00",  #           0
    b"\x66\x66\x66\x06",  # + 107374182
    b"\x00\x00\x00\x40",  # +1073741824
    b"\xff\xff\xff\x7f",  # +2147483647
]
TEST_SAMPLE_B4 = b"".join(TEST_SAMPLE_B4)

int.from_bytes(b"\x00\x80")
np.frombuffer(b"\x00\x80", "<i2")
TEST_SAMPLE = {
    Sdt.BYTES_2: TEST_SAMPLE_B2,
    Sdt.BYTES_4: TEST_SAMPLE_B4,
    Sdt.NP_I16: np.array(TEST_SAMPLE_I16, "<i2"),
    Sdt.NP_I32: np.array(TEST_SAMPLE_I32, "<i4"),
    Sdt.NP_F16: np.array(TEST_SAMPLE_F, "f2"),
    Sdt.NP_F32: np.array(TEST_SAMPLE_F, "f4"),
}


def assert_samples_almost_equal(eps: float, sample1, sample2, msg=""):
    assert all(abs(a - b) < eps for a, b in zip(sample1, sample2)), msg


@pytest.mark.parametrize(
    "dtype",
    [
        Sdt.BYTES_4,
        Sdt.NP_I32,
        Sdt.NP_F32,
    ],
)
def test_convert_to_int32_exect(dtype: Sdt):
    data_from = TEST_SAMPLE[dtype]
    data_expected = TEST_SAMPLE[Sdt.NP_I32]
    data_obtained = st.CONVERT_TO_INT32[dtype](data_from)
    assert_samples_almost_equal(4.0, data_expected, data_obtained)


@pytest.mark.parametrize(
    "dtype",
    [
        Sdt.BYTES_2,
        Sdt.NP_I16,
        Sdt.NP_F16,
    ],
)
def test_convert_to_int32_upscale(dtype: Sdt):
    data_from = TEST_SAMPLE[dtype]
    data_expected = TEST_SAMPLE[Sdt.NP_I32]
    data_obtained = st.CONVERT_TO_INT32[dtype](data_from)
    assert_samples_almost_equal(80_000.0, data_expected, data_obtained)


@pytest.mark.parametrize(
    "dtype",
    [
        Sdt.BYTES_4,
        Sdt.NP_I32,
        Sdt.NP_F32,
    ],
)
def test_convert_from_int32_exect(dtype: Sdt):
    data_from = TEST_SAMPLE[Sdt.NP_I32]
    data_expected = TEST_SAMPLE[dtype]
    data_obtained = st.CONVERT_FROM_INT32[dtype](data_from)
    assert_samples_almost_equal(4.0, data_expected, data_obtained)


@pytest.mark.parametrize(
    "dtype",
    [
        Sdt.BYTES_2,
        Sdt.NP_I16,
        Sdt.NP_F16,
    ],
)
def test_convert_from_int32_downscale(dtype: Sdt):
    data_from = TEST_SAMPLE[Sdt.NP_I32]
    data_expected = TEST_SAMPLE[dtype]
    data_obtained = st.CONVERT_FROM_INT32[dtype](data_from)
    assert_samples_almost_equal(2.0, data_expected, data_obtained)
