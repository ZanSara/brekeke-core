from typing import Union
from pathlib import Path
from subprocess import Popen

import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
from espnet2.bin.tts_inference import Text2Speech as Text2SpeechModel


class Speaker:
    
    def __init__(self, model: Union[str, Path] = "espnet/kan-bayashi_ljspeech_vits"):
        self.model = Text2SpeechModel.from_pretrained(model)

    def say(self, text: str, audio_format: str = "wav") -> Path:
        file_path = "reply.wav"

        prediction = self.model(text)
        if not prediction:
            raise ValueError(
                "The model returned no predictions. Make sure you selected a valid text-to-speech model."
            )
        output = prediction.get("wav", None)
        if output is None:
            raise ValueError(
                f"The model returned no output under the 'wav' key. "
                f"The available output keys are {prediction.keys()}. Make sure you selected the right key."
            )
        audio_data = output.cpu().numpy()

        if audio_format.upper() in sf.available_formats().keys():
            sf.write(
                data=audio_data,
                file=file_path,
                format="wav",
                subtype="PCM_16",
                samplerate=self.model.fs,
            )
        Popen(["aplay", file_path])
        sentence = AudioSegment.from_wav(file_path)
        return len(sentence)


if __name__ == "__main__":
    Speaker().say("This is a test.")
    print("Done")