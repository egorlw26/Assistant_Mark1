import os
import time
import playsound
import speech_recognition as sr
from gtts import gTTS

def speak(message):
    tts = gTTS(text = message, lang="en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)

def hear():
    r = sr.Recognizer()
    with sr.Microphone() as mic:
        voiceMessage = r.listen(mic)
        returnMessage = str()

        try:
            returnMessage = r.recognize_google(voiceMessage)
            print(returnMessage)
        except Exception as e:
            print("Exception: " + str(e))
    
    return returnMessage

audio = hear()

if "hello" in audio:
    speak("Good day!")
if "how are you" in audio:
    speak("fine, and you?")