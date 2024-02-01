from dataclasses import asdict, dataclass, replace

import noisereduce  # can take quite some time
from torch.cuda import is_available as is_cuda_available

from speech2text.audio_data import NpData


@dataclass(frozen=True)
class NoiseReduceParameters:
    stationary: bool = False
    prop_decrease: float = 0.95  # in [0.0, 1.0]
    freq_mask_smooth_hz: int = 500
    time_mask_smooth_ms: int = 50
    tmp_folder: str | None = None
    chunk_size: int = 600000
    padding: int = 30000
    n_fft: int = 1024
    clip_noise_stationary: bool = True
    use_tqdm: bool = False
    n_jobs: int = 1
    use_torch: bool = False
    device: str = "cuda"

    def as_dict(self):
        return asdict(self)

    def replace(self, **changes):
        return replace(self, **changes)


DEFAULT_NR_PARAMETERS = None
if is_cuda_available():
    DEFAULT_NR_PARAMETERS = NoiseReduceParameters().replace(
        use_torch=True, n_jobs=1
    )  # use only one CPU core if CUDA is present
else:
    DEFAULT_NR_PARAMETERS = NoiseReduceParameters().replace(
        use_torch=False, n_jobs=-1
    )  # use all the CPU cores if CUDA is absent


def reduce_noise(
    np_data: NpData,
    params: NoiseReduceParameters = DEFAULT_NR_PARAMETERS,
) -> NpData:
    new_data = noisereduce.reduce_noise(
        np_data._data,
        np_data.pcm_params.frame_rate,
        **params.as_dict(),
    )
    return NpData(np_data.pcm_params, new_data)
