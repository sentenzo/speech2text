from dataclasses import dataclass, field
from typing import Any

import numpy as np

from speech2text.transcriber.refiner import IdentityRefiner, Refiner
from speech2text.transcriber.transcriber import DummyTranscriber, Transcriber
from speech2text.utils.sample_types import SampleDType as Sdt
from speech2text.utils.samples import Samples

DEFAULT_TRANSCRIBER_CLASS = DummyTranscriber


@dataclass
class TranscriptionBlock:
    transcription: str = ""
    samples_original: Samples | None = None
    samples_refined: Samples | None = None

    def __post_init__(self):
        self.samples_original = self.samples_original or Samples()
        self.samples_refined = self.samples_refined or Samples()


class WorkflowQueue:
    def __init__(self, transcriber: Transcriber = None) -> None:
        self.text_finalized = []
        self.current_block = TranscriptionBlock()
        self.refiner = IdentityRefiner()
        self.transcriber = transcriber or DEFAULT_TRANSCRIBER_CLASS()

    def set_refiner(self, refiner: Refiner) -> "WorkflowQueue":
        self.refiner = refiner
        return self

    def set_transcriber(self, transcriber: Transcriber) -> "WorkflowQueue":
        self.transcriber = transcriber
        return self

    @staticmethod
    def _split_silance(samples: Samples) -> list[Samples]:
        ...
        # dummy
        limit = 15_000
        if len(samples) > limit:
            data: np.array = samples.as_type(Sdt.NP_I32)
            head = Samples(data[0:limit], Sdt.NP_I32)
            tail = Samples(data[limit:], Sdt.NP_I32)
            return [head, tail]
        else:
            return []

    def _update_t_block(
        self, t_block: TranscriptionBlock
    ) -> TranscriptionBlock:
        refine = self.refiner.refine
        transcribe = self.transcriber.transcribe
        t_block.samples_refined = refine(t_block.samples_original)
        t_block.transcription = transcribe(t_block.samples_refined)

    def push_chunk(self, chunk: Samples | Any, dtype: Sdt = None):
        chunk = Samples(chunk, dtype)
        samples_original = self.current_block.samples_original
        samples_original.extend(chunk)

        split_silance = WorkflowQueue._split_silance(samples_original)
        if len(split_silance) > 1:
            next_t_block = TranscriptionBlock()
            for samples in split_silance[1:]:
                next_t_block.samples_original.extend(samples)

            self.current_block.samples_original = split_silance[0]
            self._update_t_block(self.current_block)
            self.text_finalized.append(self.current_block.transcription)

            self.current_block = next_t_block

        self._update_t_block(self.current_block)

    def flush_text(self) -> list[str]:
        text = self.text_finalized[:]
        self.text_finalized = []
        return text

    def __str__(self) -> str:
        lines = self.text_finalized[:]
        lines.append(self.current_block.transcription)
        return "\n".join(lines)
