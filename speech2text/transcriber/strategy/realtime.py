from pydub.generators import WhiteNoise

from speech2text.audio_data import WHISPER_PCM_PARAMS, NpData, PdData
from speech2text.audio_data.wave_data import WavData
from speech2text.settings import (
    PyDubSettings,
    PyDubSplitOnSilenceSettings,
    WhisperSettings,
    app_settings,
)

from ..noisereduce import reduce_noise
from ..state import Block, State, Status
from ..whisper import transcribe
from .strategy import IStrategy


class RealtimeProcessing(IStrategy):
    def cold_start(self):
        temp_pcm_params = WHISPER_PCM_PARAMS
        temp_state = State(temp_pcm_params)

        def get_white_noise_block(duration_msec=1000):
            noise_wav = WavData(temp_pcm_params)
            noise_wav.append_chunk(
                WhiteNoise().to_audio_segment(duration=duration_msec).raw_data
            )
            noise_seg = PdData.load_from_wav_file(noise_wav)
            noise_arr = NpData.load_from_wav_file(noise_wav)
            return Block(noise_wav, noise_seg, noise_arr)

        self.process_chunk(
            temp_state, get_white_noise_block().raw_data.raw_data
        )

        # In order to run finalization process:
        temp_state.to_be_finalized = [get_white_noise_block()]
        temp_state.ongoing = get_white_noise_block()
        self._transcribe(temp_state)

    def process_chunk(
        self,
        state: State,
        chunk: bytes | bytearray,
        latency_ratio: float = 0.0,
    ) -> State:
        """
        ```
        +> FINALIZED
        |  ↓  increment()
        |  INCREMENTED
        |  ↓  adjust()
        |  ADJUSTED
        |  ↓  split() ------+
        |  SPLITTED         |
        |  ↓  refine()      ↓
        |  REFINED       SKIPPED
        |  ↓  transcribe()  |
        |  TRANSCRIBED      |
        +---- finalize() ←--+
        ```
        """
        state.status = Status.FINALIZED  ###############################
        state.ongoing.raw_data.append_chunk(chunk)
        state.latency_ratio = latency_ratio
        state.status = Status.INCREMENTED  #############################
        state = self._adjust(state)
        state.status = Status.ADJUSTED  ################################
        state = self._split(state)
        if state.ongoing.seg_data is None:
            state.status = Status.SKIPPED  #############################
        else:
            state.status = Status.SPLITTED  ############################
            state = self._refine(state)
            state.status = Status.REFINED  #############################
            state = self._transcribe(state)
            state.status = Status.TRANSCRIBED  #########################

        return state

    def _adjust(self, state: State) -> State:
        settings = app_settings.transcriber.stages.adjust.pydub
        seg_data = PdData.load_from_wav_file(state.ongoing.raw_data)
        if settings.low_pass_filter:
            seg_data = seg_data.low_pass_filter(settings.low_pass_filter)
        if settings.high_pass_filter:
            seg_data = seg_data.high_pass_filter(settings.high_pass_filter)
        if settings.speed_up and settings.speed_up > 1.0:
            seg_data = seg_data.speedup(settings.speed_up)
        if settings.volume_up:
            seg_data += settings.volume_up
        if settings.normalize:
            seg_data = seg_data.normalize()
        state.ongoing.seg_data = seg_data
        return state

    def _split(self, state: State) -> State:
        settings = app_settings.transcriber.stages.split
        threshold = settings.agressive_threshold
        agressive = (
            state.latency_ratio > threshold.latency_ratio
            or state.ongoing.seg_data.duration_seconds > threshold.duration_sec
        )
        if agressive:
            for preset_name, preset_params in settings.pydub_split_on_silence:
                state = self._apply_split(state, preset_params)
                if (
                    state.ongoing.seg_data is None
                    or len(state.to_be_finalized) > 0
                ):
                    break
            else:
                state = self._apply_force_split(state)
        else:
            state = self._apply_split(
                state, settings.pydub_split_on_silence.default
            )
        return state

    def _apply_split(
        self, state: State, split_params: PyDubSplitOnSilenceSettings
    ) -> State:
        init_length = len(state.ongoing.seg_data)
        segments = state.ongoing.seg_data.split_on_silence(
            **split_params.model_dump()
        )
        if len(segments) == 0:  # only silence was found
            state.to_be_finalized = []
            state.ongoing.seg_data = None
            state.ongoing.raw_data = WavData(state.input_pcm_params)
        elif len(segments) == 1:  # no splitting, but maybe trimming
            segment = segments[0]
            trim_threshold = max(
                split_params.min_silence_len - split_params.keep_silence // 2,
                200,
            )
            if (
                init_length - len(segment) > trim_threshold
            ):  # it was definitely trimmed
                state.to_be_finalized = [Block.load_from_seg_data(segment)]
                empty_wav = WavData(
                    state.input_pcm_params, b"\x00\x00\x00\x00"
                )
                empty_seg = PdData.load_from_wav_file(empty_wav)
                state.ongoing = Block(empty_wav, empty_seg)
            else:
                state.to_be_finalized = []
                state.ongoing.seg_data = segment
        elif len(segments) > 1:
            blocks = [
                Block.load_from_seg_data(segment) for segment in segments
            ]
            state.to_be_finalized = blocks[:-1]
            state.ongoing = blocks[-1]
        return state

    def _apply_force_split(self, state: State) -> State:
        l = len(state.ongoing.seg_data)

        # cut in half:
        left = state.ongoing.seg_data[: l // 2]
        right = state.ongoing.seg_data[l // 2 :]

        state.to_be_finalized = [
            Block.load_from_seg_data(left, state.input_pcm_params)
        ]
        state.ongoing.seg_data = right
        return state

    def _refine(self, state: State) -> State:
        settings_final = app_settings.transcriber.stages.refine.final

        def apply_pydub(block: Block, pydub_params: PyDubSettings):
            if pydub_params.low_pass_filter:
                block.seg_data = block.seg_data.low_pass_filter(
                    pydub_params.low_pass_filter
                )
            if pydub_params.high_pass_filter:
                block.seg_data = block.seg_data.high_pass_filter(
                    pydub_params.high_pass_filter
                )
            if pydub_params.volume_up:
                block.seg_data += pydub_params.volume_up
            if pydub_params.speed_up:
                block.seg_data = block.seg_data.speedup(pydub_params.speed_up)
            if pydub_params.normalize:
                block.seg_data = block.seg_data.normalize()

        # to_be_finalized
        if settings_final.pydub:
            for block in state.to_be_finalized:
                apply_pydub(block, settings_final.pydub)
        for block in state.to_be_finalized:
            block.arr_data = NpData.load_from_pd_data(
                block.seg_data.adjust_pcm_params(WHISPER_PCM_PARAMS)
            )
        if settings_final.noisereduce:
            for block in state.to_be_finalized:
                block.arr_data = reduce_noise(block.arr_data)

        # ongoing
        settings_ongoing = app_settings.transcriber.stages.refine.ongoing
        if settings_ongoing.pydub:
            apply_pydub(state.ongoing, settings_ongoing.pydub)
        state.ongoing.arr_data = NpData.load_from_pd_data(
            state.ongoing.seg_data.adjust_pcm_params(WHISPER_PCM_PARAMS)
        )
        if settings_ongoing.noisereduce:
            state.ongoing.arr_data = reduce_noise(state.ongoing.arr_data)

        return state

    def _transcribe(self, state: State) -> State:
        settings_final = app_settings.transcriber.stages.transcribe.final

        def apply_whisper(
            block: Block, whisper_params: WhisperSettings, **kwargs
        ):
            whisper_params = whisper_params.model_dump()
            whisper_params.update(**kwargs)
            whisper_output = transcribe(block.arr_data, **whisper_params)
            block.text = whisper_output["text"]

        for block in state.to_be_finalized:
            apply_whisper(
                block, settings_final.whisper, condition_on_previous_text=True
            )
            state.finalized.append(block)
        state.to_be_finalized = []

        settings_final = app_settings.transcriber.stages.transcribe.ongoing
        if len(state.finalized) > 0:
            state.ongoing_init_prompt = state.finalized[-1].text

        apply_whisper(
            state.ongoing,
            settings_final.whisper,
            condition_on_previous_text=False,
            initial_prompt=state.ongoing_init_prompt,
        )

        return state
