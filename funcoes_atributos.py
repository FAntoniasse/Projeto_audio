Created on Tue Jul 24 11:49:27 2018

@author: Fabrício
"""


import numpy as np
import numpy




def suave_verifica(dados,janela):
    data=dados
    tamanho = len(data)
    partes = int(tamanho/janela)
    maximo = max(data)
    corte = maximo*0.15
    result=[0]*tamanho
    for i in range(partes-1):
        media = np.mean(data[i*janela:(i+1)*janela])
        dev = np.std(data[i*janela:(i+1)*janela])
        if media < corte:
            if dev <= 0.0001:
                #result[i*100:(i+1)*100]=0
                pass
            else:
                result[i*janela:(i+1)*janela]=data[i*janela:(i+1)*janela]
        else:
            result[i*janela:(i+1)*janela]=data[i*janela:(i+1)*janela]
    return result



def corte_esq(data):

    for j in range(len(data)):    
       if(data[j]!=0):
           return j
       
        

def picos(data):
    aux_pico=[]
    aux_vale=[]
    i=0
    j=0
    inicio = data[0]
    prox = data[1]
    try:
        while (i < len(data)-1):
            if data[i+1] > data[i]:
                while data[i+1] > data[i]:
                    i += 1
                    j=0
                if (j==0 and data[i-2]!=data[i-3]):
                    aux_pico.append(i)
                j=0    
            if data[i+1] < data[i]:
                while data[i+1] < data[i]:
                    i += 1
                    j=0
                if (j==0 and data[i-2]!=data[i-3]):
                    aux_vale.append(i)
                j=0
            
            if data[i+1] == data[i]:
                    i+= 1
                    j=1
        return aux_pico, aux_vale,i
    
    
    except:
        
        return aux_pico, aux_vale,i
    
def numero_picos(data,base):
    if len(data[0])== len(data[1]):
        try:
            if data[0][0]> data[1][0]:
                pics = np.hstack((data[0]))
                u_dist = np.mean(pics)
                std_dist = np.std(pics)
                u_val = np.mean(np.asarray(base)[pics])
                std_val = np.std(np.asarray(base)[pics])
                return len(data[0]), u_dist,u_val,std_dist,std_val
            if data[0][0]< data[1][0]:
                pics = np.hstack((data[0]))
                u_dist = np.mean(pics)
                std_dist = np.std(pics)
                u_val = np.mean(np.asarray(base)[pics])
                std_val = np.std(np.asarray(base)[pics])
                return len(data[0]), u_dist,u_val,std_dist,std_val
        except:
            return 1,max(base),len(base)/2,0,0
            
        
        
        
    if len(data[0])> len(data[1]):
        u_dist = np.mean(data[0])
        std_dist = np.std(data[0])
        u_val = np.mean(np.asarray(base)[data[0]])
        std_val = np.std(np.asarray(base)[data[0]])
        return len(data[0]), u_dist,u_val,std_dist,std_val
    
    
    
    
    
    
    if len(data[0])< len(data[1]):
        if len(data[0])>0:
            pics = np.hstack((data[0]))
        if len(data[0])==0:
            pics = np.hstack((0,len(base)-1))
        u_dist = np.mean(pics)
        std_dist = np.std(pics)
        u_val = np.mean(np.asarray(base)[pics])
        std_val = np.std(np.asarray(base)[pics])
        return len(data[1]) + 1,u_dist,u_val,std_dist,std_val
    
    
    
    
    



## Retirado do git -->> https://github.com/tyiannak/pyAudioAnalysis/blob/master/audioFeatureExtraction.py
eps = 0.00000001

def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = numpy.sum(numpy.abs(numpy.diff(numpy.sign(frame)))) / 2
    return (numpy.float64(countZ) / numpy.float64(count-1.0))


def stEnergy(frame):
    """Computes signal energy of frame"""
    return numpy.sum(frame ** 2) / numpy.float64(len(frame))


def stEnergyEntropy(frame, numOfShortBlocks=10):
    """Computes entropy of energy"""
    Eol = numpy.sum(frame ** 2)    # total frame energy
    L = len(frame)
    subWinLength = int(numpy.floor(L / numOfShortBlocks))
    if L != subWinLength * numOfShortBlocks:
            frame = frame[0:subWinLength * numOfShortBlocks]
    # subWindows is of size [numOfShortBlocks x L]
    subWindows = frame.reshape(subWinLength, numOfShortBlocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)

    # Compute entropy of the normalized sub-frame energies:
    Entropy = -numpy.sum(s * numpy.log2(s + eps))
    return Entropy

def stSpectralEntropy(X, numOfShortBlocks=10):
    """Computes the spectral entropy"""
    L = len(X)                         # number of frame samples
    Eol = numpy.sum(X ** 2)            # total spectral energy

    subWinLength = int(numpy.floor(L / numOfShortBlocks))   # length of sub-frame
    if L != subWinLength * numOfShortBlocks:
        X = X[0:subWinLength * numOfShortBlocks]

    subWindows = X.reshape(subWinLength, numOfShortBlocks, order='F').copy()  # define sub-frames (using matrix reshape)
    s = numpy.sum(subWindows ** 2, axis=0) / (Eol + eps)                      # compute spectral sub-energies
    En = -numpy.sum(s*numpy.log2(s + eps))                                    # compute spectral entropy

    return En


def stHarmonic(frame, fs):
    """
    Computes harmonic ratio and pitch
    """
    M = int(numpy.round(0.016 * fs)) - 1
    R = numpy.correlate(frame, frame, mode='full')

    g = R[len(frame)-1]
    R = R[len(frame):-1]

    # estimate m0 (as the first zero crossing of R)
    [a, ] = numpy.nonzero(numpy.diff(numpy.sign(R)))

    if len(a) == 0:
        m0 = len(R)-1
    else:
        m0 = a[0]
    if M > len(R):
        M = len(R) - 1

    Gamma = numpy.zeros((M), dtype=numpy.float64)
    CSum = numpy.cumsum(frame ** 2)
    Gamma[m0:M] = R[m0:M] / (numpy.sqrt((g * CSum[M:m0:-1])) + eps)

    ZCR = stZCR(Gamma)

    if ZCR > 0.15:
        HR = 0.0
        f0 = 0.0
    else:
        if len(Gamma) == 0:
            HR = 1.0
            blag = 0.0
            Gamma = numpy.zeros((M), dtype=numpy.float64)
        else:
            HR = numpy.max(Gamma)
            blag = numpy.argmax(Gamma)

        # Get fundamental frequency:
        f0 = fs / (blag + eps)
        if f0 > 5000:
            f0 = 0.0
        if HR < 0.1:
            f0 = 0.0

    return (HR, f0)
############################################################


