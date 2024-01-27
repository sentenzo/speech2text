from pydub import AudioSegment, effects, silence

import speech2text.config as cfg
from speech2text.transcriber.samples import Samples, SamplesFormat
from speech2text.transcriber.state import TranscriptionBlock as Block
from speech2text.transcriber.state import TranscriptionState as State

from . import Transformation

SPEEDUP_RATIO = 1.3


class PdNormalize(Transformation):
    def __init__(
        self,
        amplify: float = 1.0,
    ) -> None:
        self.amplify = amplify

    def change_state(self, state: State):
        for block in state.blocks:
            audio: AudioSegment = block.samples.as_pd_a_segment()
            audio = effects.normalize(audio)
            if self.amplify != 1.0:
                audio += self.amplify
            block.samples.data = audio


class PdSplitSilence(Transformation):
    def __init__(
        self,
        min_silence_len: int = cfg.SILENCE_MIN_LEN_MSEC,
        silence_thresh: int = cfg.SILENCE_THRESHOLD_DBFS,
    ) -> None:
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh

    def change_state(self, state: State):
        assert len(state.blocks) >= 1
        for block in state.blocks:
            block.samples.as_pd_a_segment()
        block = state.blocks.pop()
        audio: AudioSegment = block.samples.as_pd_a_segment()
        segments = silence.split_on_silence(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            keep_silence=100,
            seek_step=50,  # default was 1 ms
        )
        if len(segments) > 1:
            for segment in segments:
                state.blocks.append(
                    Block(
                        samples=Samples(
                            data=segment,
                            pcm_params=state.pcm_params,
                            sample_format=SamplesFormat.PD_A_SEGMENT,
                        ),
                    )
                )
            # state.blocks[-1].init_data_b2 = block.init_data_b2
            state.latency_ratio = 0.0  # we assume that's enough to speed
        else:
            state.blocks.append(block)


class PbIfLatency_ForceSplit(Transformation):
    def __init__(
        self,
        ratio_threshold: float = cfg.FORCE_SPLIT_RATIO_THRESHOLD,
        sec_threshold: float = cfg.FORCE_SPLIT_SEC_THRESHOLD,
    ) -> None:
        self.ratio_threshold = ratio_threshold
        self.sec_threshold = sec_threshold

    def change_state(self, state: State):
        assert len(state.blocks) >= 1
        block = state.blocks[-1]
        if block.samples.len_sec() < self.sec_threshold:
            return
        if state.latency_ratio < self.ratio_threshold:
            return
        for block in state.blocks:
            block.samples.as_pd_a_segment()
        block = state.blocks.pop()
        audio: AudioSegment = block.samples.as_pd_a_segment()

        msec_padding = int(self.sec_threshold * 1000 - 1)
        pre_block = Block(
            samples=Samples(
                audio[:-msec_padding],
                pcm_params=state.pcm_params,
                sample_format=SamplesFormat.PD_A_SEGMENT,
            )
        )
        block.samples.data = audio[-msec_padding:]

        state.blocks.append(pre_block)
        state.blocks.append(block)
        state.latency_ratio = 0.0


class PdIfLatency_SpeedUp(Transformation):
    def __init__(
        self,
        ratio_threshold: float = cfg.SPEEDUP_RATIO_THRESHOLD,
        sec_threshold: float = cfg.SPEEDUP_RATIOSEC_THRESHOLD,
    ) -> None:
        self.ratio_threshold = ratio_threshold
        self.sec_threshold = sec_threshold

    def change_state(self, state: State):
        assert len(state.blocks) >= 1
        block = state.blocks[-1]
        if block.samples.len_sec() < self.sec_threshold:
            return
        if state.latency_ratio < self.ratio_threshold:
            return
        for block in state.blocks:
            block.samples.as_pd_a_segment()
        block = state.blocks.pop()
        audio: AudioSegment = block.samples.as_pd_a_segment()
        audio = effects.speedup(audio, SPEEDUP_RATIO)
        block.samples.data = audio
        state.blocks.append(block)
        state.latency_ratio = 0.0


class PdSaveWav(Transformation):
    def __init__(self, path: str) -> None:
        self.path = path

    def change_state(self, state: State):
        audio = AudioSegment.empty()
        for block in state.blocks:
            audio = audio + block.samples.as_pd_a_segment()
        audio.export(self.path, format="wav")
