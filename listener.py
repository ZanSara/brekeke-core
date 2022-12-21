import sys
import time
from random import randint
from threading import Event
from shutil import get_terminal_size

import numpy as np

from precise.runner import PreciseRunner
from precise.runner import Engine as PreciseEngine
from precise.network_runner import Listener as PreciseListener
from precise.util import buffer_to_audio, activate_notify, save_audio

from speaker import Speaker
from writer import Writer
from brain import Brain, END_TOKEN


class Listener:

    def __init__(
        self,
        model: str,
        chunk_size: int = 2048
    ):
        self.listener = PreciseListener(model, chunk_size)
        self.audio_buffer = np.zeros(self.listener.pr.buffer_samples, dtype=float)
        self.query_audio_buffer = np.zeros(self.listener.pr.buffer_samples, dtype=float)

        self.wake_engine = PreciseEngine(chunk_size=2048)
        self.wake_engine.get_prediction = self.get_wake_prediction
        self.wake_runner = PreciseRunner(
            self.wake_engine, 
            trigger_level=3,
            sensitivity=0.5,
            on_activation=self.wake, 
            on_prediction=self.aknowledge
        )
        self.query_engine = PreciseEngine(chunk_size=2048)
        self.query_engine.get_prediction = self.get_query_prediction
        self.query_runner = PreciseRunner(
            self.query_engine, 
            trigger_level=10, 
            sensitivity=0.5,
            on_activation=self.query, 
            on_prediction=self.aknowledge_query,
            trigger_immediately=True
        )
        
        self.stage = "wake"
        self.speaker = Speaker()
        self.writer = Writer()
        self.brain = Brain()
        self.in_conversation = False

        self.session_id = "%09d" % randint(0, 999999999)
        self.chunk_num = 0

    def get_wake_prediction(self, chunk):
        audio = buffer_to_audio(chunk)
        self.audio_buffer = np.concatenate((self.audio_buffer[len(audio) :], audio))
        return self.listener.update(chunk)
    
    def get_query_prediction(self, chunk):
        audio = buffer_to_audio(chunk)
        self.audio_buffer = np.concatenate((self.audio_buffer[len(audio) :], audio))
        self.query_audio_buffer = np.concatenate((self.query_audio_buffer, audio))
        # Threshold is 0.01
        value = 0.02 / np.amax(self.audio_buffer)
        print(value)
        return value
    
    def aknowledge(self, conf):
        max_width = 80
        width = min(get_terminal_size()[0], max_width)
        units = int(round(conf * width))
        bar = 'X' * units + '-' * (width - units)
        cutoff = round((1.0 - self.wake_runner.detector.sensitivity) * width)
        print(bar[:cutoff] + bar[cutoff:].replace('X', '|'))

    def aknowledge_query(self, conf):
        max_width = 80
        width = min(get_terminal_size()[0], max_width)
        units = int(round(conf * width))
        bar = 'O' * units + '.' * (width - units)
        cutoff = round((1.0 - self.query_runner.detector.sensitivity) * width)
        output = bar[:cutoff] + bar[cutoff:].replace('O', '|')
        print(output[:width])

    def wake(self):
        activate_notify()
        print("--------- TRIGGER ---------")

        self.stage = "query"
        self.wake_runner.pause()
        self.query_runner.play()

    def query(self):
        print("--------- RECEIVED ---------")
        self.query_runner.stop()
        save_audio("query.wav", self.query_audio_buffer)
        self.query_audio_buffer = np.zeros(self.listener.pr.buffer_samples, dtype=float)

        question = self.writer.transcribe("query.wav")
        if question:
            self.in_conversation = True
            
            reply, end_of_conversation = self.brain.reply(question)
            
            if reply:
                self.speaker.say(reply) / 1000
                self.query_runner.start()

            if end_of_conversation:
                self.stage = "wake"
                self.in_conversation = False
                self.query_runner.pause()
                self.wake_runner.play()            
  
        else:
            if not self.in_conversation:
                self.speaker.say("Did you call me?")
                self.query_runner.play()
            else:                
                self.stage = "wake"
                self.in_conversation = False
                self.query_runner.pause()
                self.wake_runner.play()

    def listen(self):
        try:
            self.query_runner.start()
            self.query_runner.pause()
            self.wake_runner.start()
            Event().wait()
        except KeyboardInterrupt:
            print()



if __name__ == "__main__":
    Listener(model=sys.argv[1]).listen()
