# %%
from __future__ import division
from typing_extensions import Self

from transformers import MarianTokenizer, TFMarianMTModel, pipeline
import tensorflow as tf
import torch
torch.cuda.empty_cache()
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pandas as pd
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import os
from fairseq import utils
import librosa
import soundfile as sf
import webrtcvad 
import numpy as np
import re
import sys
import threading
import zhtts
#from deepspeech import Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from six.moves import queue


# torchaudio.set_audio_backend("sox_io")
import time



from pydub import AudioSegment
from scipy.signal import butter, lfilter
# %matplotlib inline
from time import sleep, perf_counter
from threading import Thread
from glob import glob
import playsound
import librosa
import io
import wave
import subprocess
import shlex
import pyaudio

from speechbrain.pretrained import EncoderClassifier

os.environ["CUDA_VISIBLE_DEVICES"]="0"

root_path=os.getcwd()
#loading TTS model
tts = zhtts.TTS()

#loading ASR MODEL
processor_asr = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
model_asr = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

#loading Translation model
nlp=pipeline('translation_en_to_zh',tokenizer="Helsinki-NLP/opus-mt-en-zh",model="Helsinki-NLP/opus-mt-en-zh",device=0)


#loading language detection model
classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="VAD/",run_opts={device})




#translation function
def translator_model(sample_text):
    print("English: "+sample_text)
    gen=nlp(sample_text)
    return gen  



#function converts audiofile to array for ASR model
def speech_file_to_array_fn(audio,resampling_to=16000):

    speech_array, sampling_rate = torchaudio.load(audio)
    resampler=torchaudio.transforms.Resample(sampling_rate, resampling_to)
    speech = resampler(speech_array).numpy()
    return speech


#function calls the ASR model
def asr_model(audio):
    speech=speech_file_to_array_fn(audio)
    inputs = processor_asr(speech.squeeze(), sampling_rate=16_000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model_asr(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)


    results=processor_asr.batch_decode(predicted_ids)
    return results



RATE = 16000  #sampling rate
CHUNK = 160   #chunk size for audio input from microphone
TH_pool = []  #array of threads
path_r="/home/saad/speech_translation/recording/"     #directory for saving recording
path_s="/home/saad/speech_translation/synthesized/"   #directory for saving syntheized audio

#class for inference that will be used by threads
class ModelInfer(threading.Thread):
    def __init__(self, filepath):
        super(ModelInfer,self).__init__()
        self.filepath = filepath
        
    def infer(self):
        filenamed=self.filepath
       
        if os.path.exists(filenamed):
            try:
                # print(filenamed)
                out_prob, score, index, text_lab = classifier.classify_file(filenamed)  
                #print("Detected Language "+str(text_lab[0].split("_")[0].lower()))
                
                if text_lab[0].split("_")[0].lower()=='english':       #checking for target language
                    results=asr_model(filenamed)                      #calling ASR model if target language is spoken
                    os.remove(filenamed)
                
                    if results[0].strip()!='':                       #if ASR result is not empty then continue further
                        translated=translator_model(results[0].strip())             #Translation function is being called
                        tt=translated[0]['translation_text']
                        tt2=",".join(tt).split(",")
                        print("Chinese Translation:  "+str(tt))
                        tts.text2wav(tt, path_s+filenamed.split("/")[-1])             #calls the TTS model and saves the audio in synthesized folder
                            
                else:
                    
                    print("speak English Please")
                    os.remove(filenamed)
            except Exception as e:
                print("exception")
    def run(self):
        self.infer()


#create file from bytes received from the microphone 
def makefile(data,sample_format,filename):
        audio=librosa.util.buf_to_float(b''.join(data))
        clip = librosa.effects.trim(audio, top_db= 3)   #if audio contains audio of level less that 5db its stored as empty
        sf.write(filename, clip[0], 16000)
        return filename



# main Recording Function
def recording():
    p=pyaudio.PyAudio()
    vad=webrtcvad.Vad()     #voice Activity detection
    sample_format=pyaudio.paInt16
    vad.set_mode(3)        # 3 mode is the most aggressive
    i=0
    j=0
    filenamed_list = []
    file_bytes=[]
    
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)  #initializing microphone stream
    buff=[]
    count_sil=0
   
    print("Speak")

    # main continous recording loop
    while(True):       
        data=stream.read(CHUNK,exception_on_overflow=False)  #read chunk of data
        if vad.is_speech(data,RATE):                          #detect voice activity
            buff.append(data)
            # print("VAD")
        if len(buff)>0:
            if (vad.is_speech(buf=data,sample_rate=RATE))==False:
                count_sil=count_sil+1
                if count_sil>150:                            #wait for End of sentence with silence

                    filenamed=makefile(buff,p.get_sample_size(sample_format),path_r+str(i)+'.wav')
                    # print("File Created")
                    filenamed_list.append(filenamed)     #creating file list
                    
                    buff=[]
                    i=i+1
                    count_sil=0
            if len(filenamed_list)!=0:
                if (len(filenamed_list)>j):
                    flag, th = getTHread(filenamed_list[j])
                    if flag:
                        # print('Translating Speech')
                        th.start()              #creating processing thread and start procesing
                        th.join()
                        TH_pool.append(th)
                        j += 1            



#function maintains an array of threads with max value being 3. If thread is to be created this functions the availbility of space in array and
# if space is available new thread is created other wise the processing will have to wait.
def getTHread(fileName):
    if len(TH_pool) < 3: 
        return True, ModelInfer(fileName)
    else:
        for idx, thread in enumerate(TH_pool):
            if not thread.is_alive():
                TH_pool.pop(idx)
                return True, ModelInfer(fileName)
    return False, None


#this function keeps searching the synthezied directory for new files and if file is found it plays the file.

def playz():
    while True:
        try:
    
            files=glob(path_s+"/*.wav")

            for i in sorted(files):
                playsound.playsound(str(i))
                os.remove(str(i))
        except:
            print("hello")


# Main Function
def main():
    t1=Thread(target=recording)
    t1.start()
    t2=Thread(target=playz)
    t2.start()



if __name__=='__main__':
    main()




