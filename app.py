import librosa
import tensorflow as tf
import soundfile
import gradio as gr 

import pandas as pd
import os
import random
import numpy as np  
# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

audio_files_path = 'respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files'
c_names = ['Bronchiectasis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']

# Loading Audio Files
audio_files = [] 
for file in os.listdir(audio_files_path):
    if file.endswith('.wav'): 
        audio_files.append(file)  
# convert the list to a df 
audio_files_df = pd.DataFrame(audio_files, columns=['audio_paths'])
# print(audio_files_df.iloc[0]['audio_file'])
audio_files_to_show = [
   audio_files_df.iloc[2]['audio_paths'],
   audio_files_df.iloc[0]['audio_paths'],
   audio_files_df.iloc[0]['audio_paths'],
   audio_files_df.iloc[0]['audio_paths'],
   audio_files_df.iloc[0]['audio_paths']
]

# create a gradio interface
# 0. Load models 
# 1. Audio File input
# 2. clear and Submit button
# 3. Upon submit , first preprocess the audio file using log mel and then run the outputs through the AI model 
# # 4. Output the prediction

def load_model():
    # Load the model  
    return tf.keras.models.load_model("models/lung_disease_predictor_cnn_logmel_without_data_augmentation.keras")

def preprocessing(audio_file, mode):
    # we want to resample audio to 16 kHz
    sr_new = 16000 # 16kHz sample rate
    x, sr = librosa.load(audio_file, sr=sr_new) 
    # padding sound 
    # because duration of sound is dominantly 20 s and all of sample rate is 22050
    # we want to pad or truncated sound which is below or above 20 s respectively
    max_len = 5 * sr_new  # length of sound array = time x sample rate
    if x.shape[0] < max_len:
      # padding with zero
      pad_width = max_len - x.shape[0]
      x = np.pad(x, (0, pad_width))
    elif x.shape[0] > max_len:
      # truncated
      x = x[:max_len]
    
    if mode == 'mfcc':
      feature = librosa.feature.mfcc(y=x, sr=sr_new)
    
    elif mode == 'log_mel':
      feature = librosa.feature.melspectrogram(y=x, sr=sr_new, n_mels=128, fmax=8000)
      feature = librosa.power_to_db(feature, ref=np.max) 

    return feature


def predict_lung_disease(audio_data):
    # Create a temporary file
    
    filename = "temp/lungs_audio.wav"  # Set your desired filename 
    soundfile.write(filename, audio_data[1],samplerate=audio_data[0])  # Save audio to file
    
    # Process the temporary audio file
    processed_audio = preprocessing(filename, 'log_mel').reshape((-1, 128, 157, 1)) 
    new_preds = model.predict(processed_audio) 
    new_classpreds = np.argmax(new_preds, axis=1)
    print(str(c_names[new_classpreds[0]]))
    return str(c_names[new_classpreds[0]])

# Gradio Interface
model = load_model() 

# have example audio files to test 

 

# Interface
iface = gr.Interface(
    fn=predict_lung_disease,
    inputs=["audio"],
    outputs="text",
    title="Lung Disease Predictor",
    description="This is a lung disease predictor that takes in an audio file and predicts the lung disease based on the audio file.",examples=
        [  
           [os.path.join(audio_files_path,audio_file)] for audio_file in audio_files_to_show
        ]
    )



iface.launch()
