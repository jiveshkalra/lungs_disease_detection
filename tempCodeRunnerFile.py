
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
audio_files_dict ={}
for file in os.listdir(audio_files_path):
    if file.endswith('.wav'):
        patient_number = file.split('_')[0]
        if patient_number not in audio_files_dict:
            audio_files_dict[patient_number] = []
        audio_files_dict[patient_number].append(file)
        audio_files.append(file)  
# convert the dictionary to a df 
audio_files_df = pd.DataFrame.from_dict(audio_files_dict, orient='index')
print(audio_files_df.head())
