from .state import TranscriptionState as State
from .state import TranscriptionStateTransformation as StateTransform
from .transformations import DEFAULT_PIPELINE


class Workflow:
    def __init__(self, pipeline: StateTransform = DEFAULT_PIPELINE) -> None:
        self.state = State()
        self.pipeline = pipeline

    def push_chunk(self, chunk, latency_ratio: float = 0.0):
        self.state.append_chunk(chunk, latency_ratio)
        try:
            self.state.processing = True
            self.state >>= self.pipeline
        finally:
            self.state.processing = False

    def flush_finalized_text(self) -> list[str]:
        lines = self.state.text_finalized
        self.state.text_finalized = []
        return lines

    def get_transcription(self):
        lines = self.state.text_finalized[:]
        if self.state.blocks[-1].text:
            lines.append(self.state.blocks[-1].text)
        return "\n".join(lines)
