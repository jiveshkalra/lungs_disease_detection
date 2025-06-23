import gradio as gr

import soundfile



def save_audio(audio_data):

    # Assuming audio_data is the audio received from Gradio input

    filename = "recorded_audio.wav"  # Set your desired filename 
    soundfile.write(filename, audio_data[1],samplerate=audio_data[0])  # Save audio to file

    return "Audio saved as: " + filename



iface = gr.Interface(

    fn=save_audio, 

    inputs=["audio"], 

    outputs=["text"]  

)



iface.launch() 
