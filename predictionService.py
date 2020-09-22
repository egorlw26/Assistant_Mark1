import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.h5"
NUM_OF_SAMPLES_IN_1_SEC = 22050

class _Prediction_Service:

    model = None,
    _mappings = [
        "cat",
        "dog",
        "go",
        "happy",
        "left",
        "no",
        "off",
        "on",
        "sheila",
        "up",
        "wow",
        "yes"
    ]
    _instance = None

    def predict(self, file_path):
        # So, we need to convert audio file into MFCC and give it to our model
        MFCCs = self.preprocess(file_path)

        # we have now 2d dataset, but keras needs 4d (num of samples, num of segments, num of coefs, num of channels)
        # in our case there is 1 sample, 44 segments (we divided 22050 by 512), 13 coefs and 1 channel
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis] 

        predictions = self.model.predict(MFCCs)
        predicted_label = np.argmax(predictions[0])
        return self._mappings[predicted_label]

    def preprocess(self, file_path):
        
        signal, sr = librosa.load(file_path)

        if len(signal) > NUM_OF_SAMPLES_IN_1_SEC:
            signal = signal[:NUM_OF_SAMPLES_IN_1_SEC]

        MFCCs = librosa.feature.mfcc(signal, n_mfcc = 13, n_fft = 2048, hop_length=512)

        return MFCCs.T

    

def Prediction_Service():
    if _Prediction_Service._instance is None:
        _Prediction_Service._instance = _Prediction_Service()
        _Prediction_Service.model = keras.models.load_model(MODEL_PATH)
    return _Prediction_Service._instance

if __name__ == "__main__":
    pService = Prediction_Service()

    predictedWord = pService.predict("test/cat.wav")

    print(f"You said: {predictedWord}")