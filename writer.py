from pathlib import Path

try:
    import whisper
except ImportError as e:
    raise ImportError("Could not import Whisper. Run 'pip install git+https://github.com/openai/whisper.git'") from e


class Writer:

    def __init__(self):
        super().__init__()
        self.model = whisper.load_model("base.en")

    def transcribe(self, audio_file: Path, sample_rate=16000) -> str:
        return self.model.transcribe(audio_file)["text"]



if __name__ == "__main__":
    Writer().transcribe("reply.wav")