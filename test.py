import os
import time
import playsound
import speech_recognition
from gtts import gTTS

def speak(message):
    tts = gTTS(text = message, lang="en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)

speak("Hey buddy")