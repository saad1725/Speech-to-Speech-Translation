from __future__ import division
import os
os.environ["HF_DATASETS_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

from fairseq import utils
import sys
import librosa
import soundfile as sf
import wave
import webrtcvad 
import numpy as np
import pyaudio
import re
import sys
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from six.moves import queue



import time



from pydub import AudioSegment
from scipy.signal import butter, lfilter

from time import sleep, perf_counter
from threading import Thread
from glob import glob
import playsound
import librosa
import io
import wave

import pyaudio
root_path=os.getcwd()



# loading fastspeech2 model
model_eng, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/fastspeech2-en-ljspeech",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
    )
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(model_eng[0], cfg) 

#loading ASR Model    
processor_asr = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
model_asr = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")

#loading Translation model
nlp=pipeline('translation_zh_to_en',tokenizer="/home/saad/speech_translation/opus-mt-zh-en/",model="/home/saad/speech_translation/opus-mt-zh-en/",device=0)

#loading language classification model
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="VAD/",run_opts={device})


#translation function
def translator_model(sample_text):
    print("Chinese: "+sample_text)
    
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


#converts english text to speech
def tts_model_eng(text):
    model = model_eng[0]
    sample = TTSHubInterface.get_model_input(task, text)
    sample=utils.move_to_cpu(sample=sample)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return wav.unsqueeze_(0),rate

def cls():
    os.system('cls' if os.name=='nt' else 'clear')


RATE = 16000  #sampling rate
path_s='/home/saad/speech_translation/synthesized/'
def main():
    filenamed='/home/saad/speech_translation/testing_data/chinese1.mp3' #change file path here
    cls()         
    out_prob, score, index, text_lab = classifier.classify_file(filenamed)
    
    playsound.playsound(filenamed)      
    print("detected Language "+ str(text_lab[0].split("_")[0].lower()))    #checking for target language
    if text_lab[0].split("_")[0].lower()=='chinese': 
        results=asr_model(filenamed)                                       #calling ASR model if target language is spoken
        
        
        if results[0].strip()!='':                                          #if ASR result is not empty then continue further
            translated=translator_model(results[0].strip())                 #Translation model is being called
            tt=translated[0]['translation_text']
            if len(tt.split(","))>10 or len(tt.split("-"))>10:
                translated1=tt.lower().strip().split(",")
                translated2=tt.lower().strip().split("-")                    
                if (len(translated1)>7  and translated1[8].strip()==translated1[9].strip() and translated1[9].strip()==translated1[10].strip()) or (len(translated2)>7  and translated2[8].strip()==translated2[9].strip() and translated2[9].strip()==translated2[10].strip()):
                    #this condition removes the repition from the Translation model  
                
                    t1=" ".join(translated1[0:2])       
                    print("English Translation :"+str(t1))
                    wav,r=tts_model_eng(str(t1))                #calls the TTS model 
                    torchaudio.save(path_s+filenamed.split("/")[-1],wav, r) #saving the synthezied audio in the directory
                    playsound.playsound(str(path_s+filenamed.split("/")[-1]))
            else:
                print("English Translation:  "+str(tt))
                wav,r=tts_model_eng(translated[0]['translation_text']) #calls the TTS model 
                torchaudio.save(path_s+filenamed.split("/")[-1],wav, r)  #saving the synthezied audio in the directory
                playsound.playsound(str(path_s+filenamed.split("/")[-1]))


if __name__=='__main__':
    main()
