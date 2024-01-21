import noisereduce as nr
import numpy as np


class Samples:
    def __init__(
        self,
        data: bytes,
    ) -> None:
        pass


def samples_byte_to_int(
    samples: bytes | bytearray,
    sample_width: int = 2,
) -> list[int]:
    samples_int = []
    n = len(samples)
    for i in range(0, n, sample_width):
        sample = samples[i : i + sample_width]
        sample_int = int.from_bytes(
            sample,
            signed=True,
            byteorder="little",
        )
        samples_int.append(sample_int)
    return samples_int


def int_to_bytes(
    samples: list[int] | bytes | bytearray,
):
    ...


def samples_to_float(
    samples: list[int] | bytes | bytearray,
    sample_width: int = 2,
) -> np.float16 | np.float32:
    if not samples or isinstance(samples[0], np.float16 | np.float32):
        return samples
    if isinstance(samples, bytes | bytearray):
        samples = samples_byte_to_int(samples, sample_width)

    float_type = (
        None,
        None,
        np.float16,  # 2
        None,
        np.float32,  # 4
    )[sample_width]
    samples = np.array(samples).astype(float_type)
    max_int = 1 << (8 * sample_width - 1)
    samples /= max_int
    return samples


def samples_to_int(
    samples: list[int | np.float16 | np.float32] | bytes | bytearray,
    sample_width: int = 2,
) -> list[int]:
    if isinstance(samples, list):
        if not samples or isinstance(samples[0], int):
            return samples
        if isinstance(samples[0], np.float16 | np.float32):
            max_int = 1 << (8 * sample_width - 1)
            samples *= max_int
            samples = np.array(samples).astype(int)
            return samples
    if isinstance(samples, bytes | bytearray):
        return samples_byte_to_int(samples, sample_width)


def reduce_noise(samples, sample_width):
    init_samples_type = type(samples)
    samples = samples_to_float(samples, sample_width)
    samples = nr.reduce_noise(samples)
    if init_samples_type in (bytes, bytearray):
        return
