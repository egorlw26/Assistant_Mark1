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

def hearGoogleAPI():
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

def hearCustomCNN():
    r = sr.Recognizer()
    with sr.Microphone() as mic:
        voiceMessage = r.listen(mic)
    
    with open("myVoice.wav", "wb") as aFile:
        aFile.write(voiceMessage.get_wav_data())

if __name__ == "__main__":
    hearCustomCNN()

