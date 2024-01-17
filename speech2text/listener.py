import time
import wave

import pyaudio as pa


class PyAudioWrapper(pa.PyAudio):
    def __init__(self) -> None:
        super().__init__()
        self.inside_cm = False

    def __enter__(self):
        assert not self.inside_cm
        self.inside_cm = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.inside_cm = False
        self.terminate()


class Listener:
    def chunks(self, chunk_duration_sec: float):
        while True:
            time.sleep(chunk_duration_sec)

    @staticmethod
    def get_device_list():
        with PyAudioWrapper() as audio:
            device_list = []
            for i in range(audio.get_device_count()):
                device_list.append(audio.get_device_info_by_index(i))
            return device_list


class MicrophoneListener(Listener):
    def __init__(self, device_id=None) -> None:
        with PyAudioWrapper() as audio:
            self.device_info = None
            if device_id is None:
                self.device_info = audio.get_default_input_device_info()
            else:
                self.device_info = audio.get_device_info_by_index(device_id)

            self.sample_rate = int(self.device_info["defaultSampleRate"])

    def chunks(self, chunk_duration_sec: float):
        with PyAudioWrapper() as audio:
            audio.open()


class FileListener(Listener):
    def __init__(self, audio_file_path) -> None:
        self.audio_file_path = audio_file_path
        with wave.open(self.audio_file_path, "rb") as wav_file:
            self.sample_rate = wav_file.getframerate()

    def chunks(self, chunk_duration_sec: float):
        with wave.open(self.audio_file_path, "rb") as wav_file:
            (nchannels, sampwidth, framerate, _, _, _) = wav_file.getparams()
            chunk_size = int(framerate * nchannels * chunk_duration_sec)
            while True:
                chunk = wav_file.readframes(chunk_size)
                if not chunk:
                    break
                if nchannels > 1:
                    mono_chunk = chunk[::nchannels]
                    for i in range(chunk_size // nchannels):
                        for j in range(1, nchannels):
                            mono_chunk[i] += mono_chunk[i + j]
                        mono_chunk[i] /= nchannels
                    chunk = bytes(mono_chunk)
                yield chunk
            while True:
                yield b"\x00" * int(framerate * chunk_duration_sec * sampwidth)
