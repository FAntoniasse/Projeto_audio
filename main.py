import segmentacao  as sg
import os
from multiprocessing import  Pool



TRAINING_FOLDER = '/home/fabricio/Desktop/projetoaudio/base_treinamento_I'
TEST_FOLDER = '/home/fabricio/Desktop/projetoaudio/base_validacao_I'
TRAINING_OUTPUT = '/home/fabricio/Desktop/projetoaudio/output_train'
TEST_OUTPUT = '/home/fabricio/Desktop/projetoaudio/output_test'



def create_train(folder):
    
    sg.segmenta(os.path.join(TRAINING_FOLDER,folder), TRAINING_OUTPUT)


def create_test(folder):
    
    sg.segmenta(os.path.join(TEST_FOLDER,folder), TEST_OUTPUT)



def create_folder_structure():
    """Cria a estrutura de pastas
    necessaria, caso ela nao exista.
    Em seguida, faz a segmentacao.

    As pastas TEST_OUTPUT e TRAINING_OUTPUT
    contem os captchas segmentados.
    """
    if not os.path.exists(TRAINING_OUTPUT):
        os.mkdir(TRAINING_OUTPUT)
    if not os.path.exists(TEST_OUTPUT):
        os.mkdir(TEST_OUTPUT)
        
       
    wkers = Pool(8) #-> colocar a quantidade de nucleos que deseja usar
    wkers.map(create_train,os.listdir(TRAINING_FOLDER))
    wkers.close()
    wkers.join()
    
    wkers = Pool(8)
    wkers.map(create_test,os.listdir(TEST_FOLDER))
    wkers.close()
    wkers.join()
        
   

if __name__ == "__main__":
    create_folder_structure()
    #from modelo import break_captcha
    #break_captcha()
