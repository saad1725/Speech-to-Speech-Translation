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

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

RATE = 16000  #sampling rate
path_s="/home/saad/speech_translation/synthesized"
def main():
    filenamed='/home/saad/speech_translation/testing_data/english.wav'          #Change file path here
    cls()
    out_prob, score, index, text_lab = classifier.classify_file(filenamed)
    playsound.playsound(filenamed)   
    print("Detected Language "+str(text_lab[0].split("_")[0].lower()))
    if text_lab[0].split("_")[0].lower()=='english':       #checking for target language
        results=asr_model(filenamed)
                             #calling ASR model if target language is spoken
        if results[0].strip()!='':                       #if ASR result is not empty then continue further
            translated=translator_model(results[0].strip())             #Translation function is being called
            tt=translated[0]['translation_text']
            print("Chinese Translation:  "+str(tt))
            tts.text2wav(tt, path_s+filenamed.split("/")[-1])
            playsound.playsound(str(path_s+filenamed.split("/")[-1]))



if __name__=='__main__':
    main()
