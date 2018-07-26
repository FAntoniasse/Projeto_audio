
import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
from tqdm import tqdm
tqdm.pandas()
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from spectrum import get_spectrum
from math import fabs
import scipy
import numpy
from funcoes_atributos import *


TRAINING_OUTPUT = '/home/fabricio/Desktop/projetoaudio/output_train/'
TRAINING_AUDIO_CAPTCHA_FOLDERS = [TRAINING_OUTPUT+i for i in os.listdir(TRAINING_OUTPUT)]
TRAINING_AUDIO_FILENAMES = [] # -> <number>_<digit>.wav
for folder in TRAINING_AUDIO_CAPTCHA_FOLDERS:
    for f in os.listdir(folder):
        TRAINING_AUDIO_FILENAMES.append(folder+'/'+f)

TEST_OUTPUT = '/home/fabricio/Desktop/projetoaudio/output_test/'
TEST_AUDIO_CAPTCHA_FOLDERS=[]
for folder in os.listdir(TEST_OUTPUT):
    TEST_AUDIO_CAPTCHA_FOLDERS.append(TEST_OUTPUT+folder)


SAMPLE_RATE = 50000
def extract_features(audio_filename: str, path: str) -> pd.core.series.Series:
    data, _ = librosa.core.load(path + audio_filename, sr=SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        label = audio_filename.split('.')[0].split('-')[-1]
        
        
        #####Pre-proc
        data = librosa.util.normalize(data)
        janela = int(round(len(data)/4.8))
        if janela%2 == 0:
            janela += 1
        data1_positivo = list(map(fabs,data))
        ydata3 = scipy.signal.savgol_filter(data1_positivo,janela,1)
        
        janela = int(round(len(data)/1084))
        ydata3 = suave_verifica(ydata3,janela)
             
        j_esq = corte_esq(ydata3)
        j_dir = corte_esq(list(reversed(ydata3)))
        
        data = data[j_esq:len(data)-j_dir]
        #####
        
        ###############################################
        zero = stZCR(data)
        energia = stEnergy(data)
        entro = stEnergyEntropy(data)
        spect = stSpectralEntropy(data)
        har = stHarmonic(data,fs=50000)
        ###############################################
        
        
        
        
        ft1_raw = librosa.feature.mfcc(data, sr=SAMPLE_RATE, n_mfcc=40)
        ft1 = np.array([list(map(fabs, sublist)) for sublist in ft1_raw]) # Tudo positivo
        delta2 = np.max(ft1,axis=1) - np.min(ft1,axis=1)
        
        
        ft2 = librosa.feature.zero_crossing_rate(y=data)
        ft2_trunc = np.mean(ft2)    
        
        
        feature3 = librosa.feature.spectral_rolloff(data)
        feature3_flat = np.hstack((np.median(feature3), np.std(feature3)))
    
        feature4 = librosa.feature.spectral_centroid(data)
        feature4_flat = np.hstack((np.median(feature4), np.std(feature4)))
        
          
        feature7 = librosa.feature.tonnetz(data)
        feature7_flat = np.hstack((np.median(feature7), np.std(feature7)))
        
        
        mel_raw = librosa.feature.melspectrogram(y=data)[:20,:]
        mdelta2 = np.max(mel_raw,axis=1) - np.min(mel_raw,axis=1)
        
        
        
        
        
        
        
        
        features = pd.Series(np.hstack((zero,energia,entro,spect, mdelta2,ft2_trunc,
                                        get_spectrum(data),delta2,  feature3_flat,
                                        feature4_flat,feature7_flat,label)))
        
        
        return features
    except:
        print(path + audio_filename)
        return pd.Series([0]*79)


def train() -> tuple:
    X_train_raw = []
    y_train = []
    for sample in TRAINING_AUDIO_FILENAMES:
        folder = '/home/fabricio/Desktop/projetoaudio/output_train/'+sample.split('/')[-2]+'/'
        filename = sample.split('/')[-1]
        obj = extract_features(filename, folder)
        d = obj[0:obj.size - 1]
        l = obj[obj.size - 1]
        X_train_raw.append(d)
        y_train.append(l)

    # Normalise
    std_scale = preprocessing.StandardScaler().fit(X_train_raw) 
    X_train = std_scale.transform(X_train_raw)
    
    return X_train, y_train, std_scale


def test(X_train: np.ndarray, y_train: np.ndarray, std_scale: preprocessing.data.StandardScaler):
    
    accuracy1NN = 0
    accuracySVM = 0

   
    
    

    neigh1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    svm = SVC().fit(X_train, y_train)
    
    
    for folder in TEST_AUDIO_CAPTCHA_FOLDERS:
        
        correct1NN = 0
        correctSVM = 0
        correctQDA = 0
        correctLDA = 0
       
        folder = folder+'/'
        
        for filename in os.listdir(folder):
            obj = extract_features(filename, folder)
            y_test = obj[obj.size - 1]
            X_test_raw = [obj[0:obj.size - 1]]
            X_test = std_scale.transform(X_test_raw) # normalise
                        
            
            y_pred = neigh1.predict(X_test)
            if y_pred[0] == y_test:
                correct1NN+=1
                
            y_pred = svm.predict(X_test)
            if y_pred[0] == y_test:
                correctSVM+=1
                
           
        
        if correct1NN == 4:
            accuracy1NN+=1
       
        if correctSVM == 4:
            accuracySVM+=1
        
            
        
    print("Acuracia 1NN = "+str(accuracy1NN / len(TEST_AUDIO_CAPTCHA_FOLDERS)))
    print("Acuracia SVM = "+str(accuracySVM / len(TEST_AUDIO_CAPTCHA_FOLDERS)))

    
    
    


def important_features() -> np.ndarray:
    """Retorna um array com as features mais importantes,
    extraidas a partir da base de treino.
    """
    X, Y, std_scale = train()
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rnd_clf.fit(X, Y)
    print(rnd_clf.feature_importances_)
    return rnd_clf.feature_importances_, X


def break_captcha():
    X_train, y_train,  std = train()
    test(X_train, y_train,  std)

if __name__ == "__main__":
    break_captcha()
    
