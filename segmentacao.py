"""
Created on Thu Jul 26 19:27:04 2018

@author: Fabrício
"""

import librosa
import librosa.display
import numpy as np
import os
import scipy
from math import fabs

def inicio(data,tamanho):
    for i in [1,2,3,4]:
        if (data[0] > i*tamanho and data[1]<(i+1)*tamanho):
            return i
    return 0

def org(letras):
    tempo = list(map(lambda x: x[1],letras))
    temp_ord = list(sorted(tempo))
    let=[]
    for i in temp_ord:
        aux = tempo.index(i)
        let.append(letras[aux][2])
        
    return let

def extract_labels(filename):
    return os.path.splitext(os.path.basename(filename))[0]


        
        
def sec(base,data,ini,ref,reg):
    
    for i in data[ini:]:
        if(abs(list(base).index(i)-ref)>reg):
            return i



def ter(base,data,ini,ref,reg):
    
    for i in data[data.index(ini):]:
        if(abs(list(base).index(i)-ref[0])>reg):
            if(abs(list(base).index(i)-ref[1])>reg):
                return i

def qrt(base,data,ini,ref,reg):
    try:
        for i in data[data.index(ini):]:
            if(abs(list(base).index(i)-ref[0])>reg):
                if(abs(list(base).index(i)-ref[1])>reg):
                    if(abs(list(base).index(i)-ref[2])>reg):
                        return i
    except:
        pass
             


def maximos(data):
    tamanho = len(data)/4
    reg = tamanho*0.7
    c=list(sorted(data,reverse=1))
    lista=[]
    primeiro = max(c)
    lista.append((primeiro,list(data).index(primeiro),data[list(data).index(primeiro)-int(reg/4):list(data).index(primeiro)+int(reg/4)]))
    
    seg = sec(data,c,0,lista[0][1],reg)
    #print(seg)
    lista.append((seg,list(data).index(seg),data[list(data).index(seg)-int(reg/4):list(data).index(seg)+int(reg/4)]))
    
    terc = ter(data,c,lista[1][0],[lista[0][1],lista[1][1]],reg)
    #print(terc)
    lista.append((terc,list(data).index(terc),data[list(data).index(terc)-int(reg/4):list(data).index(terc)+int(reg/4)]))
    
    qrat = qrt(data,c,lista[2][0],[lista[0][1],lista[1][1],lista[2][1]],reg)
    #print(qrat)
    lista.append((qrat,list(data).index(qrat),data[list(data).index(qrat)-int(reg/4):list(data).index(qrat)+int(reg/4)]))
    
    
    verifica = list(map(lambda x : len(x[2]),lista))
    
    for i in range(4):
        if verifica[i]==0:
            if lista[i][1]<(len(data)/2):
                aux = data[:lista[i][1]+int(reg/3)]
                lista[i]= (lista[i][0],lista[i][1],aux)
            else:
                aux = data[lista[i][1]-int(reg/3):]
                lista[i]=(lista[i][0],lista[i][1],aux)
    
    
    return lista




def segmenta(filename, output):
    try:
        data12, sr = librosa.core.load(filename, sr=5000)
    
        letras = maximos(data12)
        
        letras = org(letras) 
                   
        
        labels = extract_labels(filename)
        if not os.path.exists('%s/%s' % (output, labels)):
            os.mkdir('%s/%s' % (output, labels))
        for i in range(4):
            
            librosa.output.write_wav('%s/%s/%d-%s.wav' % (output, labels, i, labels[i]), letras[i], sr=sr)
    except:
        print(filename)
        pass
