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

import pyaudio

os.environ["CUDA_VISIBLE_DEVICES"]="0"
model_eng, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/fastspeech2-en-ljspeech",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
    )
processor_asr = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
model_asr = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
tokenizer = MarianTokenizer.from_pretrained("/home/saad/chinese_to_english/opus-mt-zh-en")
model_t = TFMarianMTModel.from_pretrained("/home/saad/chinese_to_english/opus-mt-zh-en",from_pt=True)
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator(model_eng[0], cfg)  
# from speechbrain.pretrained import EncoderClassifier
# classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir="/home/saad/chinese_to_english/VAD/",run_opts={device})

def translator_model(sample_text):
    print("Chinese: "+sample_text)
    nlp=pipeline('translation_zh_to_en',tokenizer="/home/saad/chinese_to_english/opus-mt-zh-en",model="/home/saad/chinese_to_english/opus-mt-zh-en",device=0)
    gen=nlp(sample_text)
    # print("English: "+str(gen[0]['translation_text']))
    return gen  

def speech_file_to_array_fn(audio,resampling_to=16000):

    speech_array, sampling_rate = torchaudio.load(audio)
    resampler=torchaudio.transforms.Resample(sampling_rate, resampling_to)
    speech = resampler(speech_array).numpy()
    return speech

def asr_model(audio):
    speech=speech_file_to_array_fn(audio)
    inputs = processor_asr(speech.squeeze(), sampling_rate=16_000, return_tensors="pt", padding=True)
    # print(inputs.input_values.shape)
    with torch.no_grad():
        logits = model_asr(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    #print("Prediction:", processor_asr.batch_decode(predicted_ids))
    results=processor_asr.batch_decode(predicted_ids)
    return results



def stt_model_eng(text):
    model = model_eng[0]

    sample = TTSHubInterface.get_model_input(task, text)
    
    sample=utils.move_to_cpu(sample=sample)
    # print(sample)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return wav.unsqueeze_(0),rate

class MicrophoneStream(object):
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk


        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,

            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,

            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
 
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

RATE = 16000
CHUNK = 160
TH_pool = []
path_r="/home/saad/chinese_to_english/recording/"
path_s="/home/saad/chinese_to_english/synthesized/"


pattern = re.compile(".(\w+)(,\w+)(,\w+).")
class ModelInfer(threading.Thread):
    def __init__(self, filepath):
        super(ModelInfer,self).__init__()
        self.filepath = filepath
    def infer(self):
        filenamed=self.filepath
        if os.path.exists(filenamed):
            try:
                results=asr_model(filenamed)
               
                
                if results[0].strip()!='':
                    translated=translator_model(results[0].strip())
                    tt=translated[0]['translation_text']
                    if len(tt.split(","))>10 or len(tt.split("-"))>10:
                        translated1=tt.lower().strip().split(",")
                        translated2=tt.lower().strip().split("-")
                        if (len(translated1)>7  and translated1[8].strip()==translated1[9].strip() and translated1[9].strip()==translated1[10].strip()) or (len(translated2)>7  and translated2[8].strip()==translated2[9].strip() and translated2[9].strip()==translated2[10].strip()):
                            # print("Recurrent 2")
                        
                            t1=" ".join(translated1[0:2])
                            print("English Translation :"+str(t1))
                            wav,r=stt_model_eng(str(t1))
                            torchaudio.save(path_s+filenamed.split("/")[-1],wav, r)
                    else:
                        print("English Translation:  "+str(tt))
                        wav,r=stt_model_eng(translated[0]['translation_text'])
                        torchaudio.save(path_s+filenamed.split("/")[-1],wav, r)
            except:
                print("processing exception")
    def run(self):
        self.infer()

def makefile(data,sample_format,filename):
        audio=librosa.util.buf_to_float(b''.join(data))
        clip = librosa.effects.trim(audio, top_db= 3)
        sf.write(filename, clip[0], 16000)
        return filename

def recording():
    p=pyaudio.PyAudio()
    vad=webrtcvad.Vad()
    sample_format=pyaudio.paInt16
    vad.set_mode(3)
    i=0
    j=0
    filenamed_list = []
    # print("starting recording")
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    buff=[]
    count_sil=0
    # step=0
    print("Speak")
    while(True):       
        data=stream.read(CHUNK,exception_on_overflow=False)
        if vad.is_speech(data,RATE):
            buff.append(data)
            # print("VAD")
        if len(buff)>0:
            if (vad.is_speech(buf=data,sample_rate=RATE))==False:
                count_sil=count_sil+1
                if count_sil>150:

                    filenamed=makefile(buff,p.get_sample_size(sample_format),path_r+str(i)+'.wav')
                    # print("File Created")
                    filenamed_list.append(filenamed)
                    buff=[]
                    i=i+1
                    count_sil=0
            if len(filenamed_list)!=0:
                if (len(filenamed_list)>j):
                    flag, th = getTHread(filenamed_list[j])
                    if flag:
                        # print('Translating Speech')
                        th.start()
                        th.join()
                        TH_pool.append(th)
                        j += 1            

def getTHread(fileName):
    if len(TH_pool) < 3: 
        return True, ModelInfer(fileName)
    else:
        for idx, thread in enumerate(TH_pool):
            if not thread.is_alive():
                TH_pool.pop(idx)
                return True, ModelInfer(fileName)
    return False, None

def playz():
    while True:
        try:
    
            files=glob(path_s+"/*.wav")

            for i in sorted(files):
                playsound.playsound(str(i))
                os.remove(str(i))
        except:
            print("hello")

def main():
    t1=Thread(target=recording)
    t1.start()
    t2=Thread(target=playz)
    t2.start()



        
    


                    

                    

# %%
if __name__=='__main__':
    main()

# %%



