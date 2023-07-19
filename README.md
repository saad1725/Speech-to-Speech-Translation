###################################################

Install the requirements using requirements.txt

Python version 3.8


##################################################

This script uses speechbrain models to perform speech to speech translation from English to Chinese and in the reverse order also.

The two modes are manual and automatic translation.


This code uses threading to perform without lag Translation.
################################################

Model used for   chinese to english translation


"opus-mt-zh-en"


ASR model for chinsese

"wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"


Model used for english to chinese translation

"Helsinki-NLP/opus-mt-en-zh"


ASR model for English

"wav2vec2-large-xlsr-53-english"

Model used for voice activity detection

"speechbrain/lang-id-commonlanguage_ecapa"


Model for speech synthesis

"facebook/fastspeech2-en-ljspeech with Hifi-gan vocoder"

#####################################################


