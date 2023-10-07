import io
import os
import pickle
import xgboost
import uuid
import numpy as np
import gradio as gr
from scipy.io.wavfile import write,read

from lib.utils import *

path_audio = 'audio'
if not os.path.exists(path_audio):
    os.mkdir(path_audio)
    
folder_model = 'model'
if not os.path.exists(folder_model):
    os.mkdir(folder_model)

path_model = 'model/model_selection'
with open(path_model, "rb") as fp:   # Unpickling
  model = pickle.load(fp)

def process_classify(input_audio,file_audio):
    if input_audio != None:
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, input_audio[0], np.array(input_audio[1]))
    elif input_audio == None and file_audio != None:
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, file_audio[0], np.array(file_audio[1]))
    elif input_audio == None and file_audio == None:
        return "Cannot Detected Cough"
   
    
    path_outpath = os.path.join('audio',str(uuid.uuid4()) + '.wav')
    outpath = to_wav(byte_io,path_outpath)
    
    sr, data = read(outpath)
    predict = classify_cough(data,sr,model)
    
    return predict

with gr.Blocks(title="Cough Detection") as demo:
    gr.Markdown("""
                # Cough Detection
                Start typing below and then click **Run** to see the output.
                """)
    with gr.Row():
        with gr.Column():
            inp = gr.Microphone()
            file_inp = gr.Audio()
        with gr.Column():
            out = gr.Label()
            with gr.Row():
                clear_btn = gr.Button("Clear",variant='secondary')
                run_btn = gr.Button("Run",variant='primary')
    
    run_btn.click(fn=process_classify, inputs=[inp,file_inp], outputs=out)
    clear_btn.click(lambda: [None,None,""], outputs=[inp,file_inp,out])
    
demo.launch()