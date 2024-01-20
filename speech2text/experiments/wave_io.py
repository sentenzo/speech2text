from __future__ import annotations

import logging
import wave
from os import PathLike
from typing import Callable

N_FRAMES = 128
logger = logging.getLogger(__name__)


class WaveTransformation:
    def __init__(
        self, transformation: Callable[[wave._wave_params, bytes], bytes]
    ) -> None:
        ...


class WaveStream:
    def __init__(self, path: PathLike) -> None:
        wave_file: wave.Wave_read
        with wave.open(path, "rb") as wave_file:
            self.wave_params = wave_file.getparams()
            logger.debug("self.wave_params = %s", self.wave_params)
        self.path = path
        self.transformations = []

    def clone(self) -> WaveStream:
        new: WaveStream = WaveStream.__new__(WaveStream)
        new.path = self.path
        new.wave_params = self.wave_params
        new.transformations = self.transformations[:]
        return new

    def set_nchannels(self, nchannels: int) -> WaveStream:
        new = self.clone()
        if self.wave_params.nchannels == nchannels:
            return new
        new.wave_params = new.wave_params._replace(nchannels=nchannels)

        sampwidth = self.wave_params.sampwidth
        frame_size = sampwidth * self.wave_params.nchannels

        def transformation(frames: bytes) -> bytes:
            t_frames = bytearray()
            for frame_ptr in range(0, len(frames), frame_size):
                frame = frames[frame_ptr : frame_ptr + frame_size]
                acc_channel = 0
                for channel_ptr in range(0, frame_size, sampwidth):
                    channel = frame[channel_ptr : channel_ptr + sampwidth]
                    acc_channel += int.from_bytes(
                        channel, signed=True, byteorder="little"
                    )
                acc_channel //= self.wave_params.nchannels
                acc_channel = acc_channel.to_bytes(
                    length=sampwidth, signed=True, byteorder="little"
                )
                if nchannels > self.wave_params.nchannels:
                    t_frames.extend(frame)
                    added_channels_count = (
                        nchannels - self.wave_params.nchannels
                    )
                    t_frames.extend(acc_channel * added_channels_count)
                elif nchannels < self.wave_params.nchannels:
                    t_frames.extend(frame[: -2 * sampwidth])
                    t_frames.extend(acc_channel)

            return bytes(t_frames)

        new.transformations.append(transformation)
        return new

    def write_all_to_file(self, out_path: PathLike) -> None:
        in_file: wave.Wave_read
        out_file: wave.Wave_write
        with wave.open(self.path, "rb") as in_file:
            with wave.open(out_path, "wb") as out_file:
                out_file.setparams(self.wave_params)
                while frames := in_file.readframes(N_FRAMES):
                    t_frames = frames
                    for tr in self.transformations:
                        t_frames = tr(t_frames)
                    out_file.writeframes(t_frames)
