from difflib import SequenceMatcher
from pathlib import Path

_path_to_audio = Path(__file__).parent / "audio_samples"

AUDIO_FILES = {
    "en_123.wav": {
        "path": (_path_to_audio / "en_123.wav").as_posix(),
        "transcription": " 1, 2, 3",
    },
    "en_chunk.wav": {
        "path": (_path_to_audio / "en_chunk.wav").as_posix(),
        "transcription": " Nudge gently but wake her now. The news struck out into the restless minds. Once we stood beside the shore, a chunk in the wall allowed a draft to blow. Fastened to pins on each side, a cold dip restores health and zest. He takes the oath of office each march. The sand drifts over the sill of the old house. The point of the steel pan was bent and twisted. There is a lag between thought and act.",
    },
    "en_news.wav": {
        "path": (_path_to_audio / "en_news.wav").as_posix(),
        "transcription": " Wanted, Chief Justice of the Massachusetts Supreme Court, in April the SJC's current leader Edward Hennessy reaches the mandatory retirement age of 70 and a successor is ex...",
    },
    "en_hedgehog.wav": {
        "path": (_path_to_audio / "en_hedgehog.wav").as_posix(),
        "transcription": " 1, 2, 3, 4. Cat, dog, hedgehog. I always go to bed when my eyes are red. What is the color of night? Something must break now.",
    },
    "en_crows.wav": {
        "path": (_path_to_audio / "en_crows.wav").as_posix(),
        "transcription": " Let me tell you about New Caledonian Crows. Now I'm not talking about the Old Caledonian Crows. Those ones were idiots. No, the New Caledonian Crows live on a group of items called New Caledonia. Must be named after the crows.",
    },
}


def seq_similarity(seq1, seq2) -> float:
    return SequenceMatcher(None, seq1, seq2).ratio()
