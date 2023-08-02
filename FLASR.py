import io
import os
import base64
import sys
import time
import warnings
import string
import numpy as np
import imageio
from copy import deepcopy
from pathlib import Path
from typing import (Iterable, List, Optional,
                    TextIO, Union)
import riva.client
import riva.client.audio_io
import riva.client.proto.riva_asr_pb2 as rasr
import streamlit as st
import wave
from streamlit_drawable_canvas import st_canvas
from pydub import AudioSegment
# import streamlit_ace as st_ace
from st_on_hover_tabs import on_hover_tabs

# ===========Grammar imports=============
from gramformer import Gramformer


# =========Suggestion Generator Imports=============
import torch
from transformers import BertTokenizer, BertForMaskedLM, top_k_top_p_filtering, logging
logging.set_verbosity_error()
#Bert Variables Declaration
no_words_to_be_predicted = globals()
select_model = globals()
enter_input_text = globals()


# # ===========Image Captioning Imports==============
# import cv2
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel

# # Load the CLIP model and processor
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")




# ==============ASR API===============
# DO NOT CHANGE THIS VALUE
uri = "ec2-3-208-22-219.compute-1.amazonaws.com:50051"

lang="en-US"

qryParam =st.experimental_get_query_params()


if "lang" in qryParam:
    if qryParam["lang"][0]:
        lang=qryParam["lang"][0]



auth = riva.client.Auth(uri=uri)
riva_tts = riva.client.SpeechSynthesisService(auth)
sample_rate_hz = 44100
req = { 
        "language_code"  : "en-US",
        "encoding"       : riva.client.AudioEncoding.LINEAR_PCM ,   # Currently only LINEAR_PCM is supported
        "sample_rate_hz" : sample_rate_hz,                          # Generate 44.1KHz audio
        "voice_name"     : "English-US.Female-1"                    # The name of the voice to generate
}

asr_service = riva.client.ASRService(auth)

offline_config = riva.client.RecognitionConfig(
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    max_alternatives=1,
    language_code = lang,
    enable_automatic_punctuation=True,
    verbatim_transcripts=False,
)
streaming_config = riva.client.StreamingRecognitionConfig(config=deepcopy(offline_config), interim_results=False)

st.set_page_config(layout="wide")
st.image("Original.png", width=100)
st.title("FirstLanguage ASR Demo")
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


riva.client.add_word_boosting_to_config(offline_config, ["1st language"], -20)
riva.client.add_word_boosting_to_config(offline_config, ["FirstLanguage"], 20)
riva.client.add_word_boosting_to_config(offline_config, ["AntiBERTa"], 10)
riva.client.add_word_boosting_to_config(offline_config, ["ABLooper"], 10)

# speech_hints = ["My $OOV_ALPHA_SEQUENCE name", "Time now is $TIME","Current time is $TIME", "It is $TIME", "It will cost $MONEY"]
# boost_lm_score = 4.0
# riva.client.add_word_boosting_to_config(offline_config, speech_hints, boost_lm_score)




PRINT_STREAMING_ADDITIONAL_INFO_MODES = ['no', 'time', 'confidence']
textReturned = ''


# ==========GRAMMAR CORRECTION==================================
def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)
gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector


# =====DEFINE FUNCTIONS FOR BERT NEXT WORD PREDICTION MODEL==========
def set_model_config(**kwargs):
  for key, value in kwargs.items():
    print("{0} = {1}".format(key, value))
  
  no_words_to_be_predicted = list(kwargs.values())[0] # integer values
  select_model = list(kwargs.values())[1] # possible values = 'bert' or 'gpt' or 'xlnet'
  enter_input_text = list(kwargs.values())[2] #only string

  return no_words_to_be_predicted, select_model, enter_input_text

def load_model(model_name):
  try:
    if model_name.lower() == "bert":
      bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
      return bert_tokenizer,bert_model
    else:
        print("tf?")
  except Exception as e:
    pass

def get_all_predictions(text_sentence,  model_name, top_clean=5):
  if model_name.lower() == "bert":
    # ========================= BERT =================================
    input_ids, mask_idx = encode_bert(bert_tokenizer, text_sentence)
    with torch.no_grad():
      predict = bert_model(input_ids)[0]
    bert = decode_bert(bert_tokenizer, predict[0, mask_idx, :].topk(no_words_to_be_predicted).indices.tolist(), top_clean)
    return {'bert': bert}

#BERT ENCODE AND DECODE
# BERT ENCODE
def encode_bert(tokenizer, text_sentence, add_special_tokens=True):
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
  # if <mask> is the last token, append a "." so that models dont predict punctuation.
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx
  
# BERT DECODE
def decode_bert(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])

def get_prediction_end_of_sentence(input_text, model_name):
  try:
    if model_name.lower() == "bert":
      input_text += ' <mask>'
      print(input_text)
      res = get_all_predictions(input_text, model_name, top_clean=int(no_words_to_be_predicted)) 
      return res
    else:
        print("Tf2?")

  except Exception as error:
    pass



# ================PRINT STREAMING=================================
def print_streaming(
    responses: Iterable[rasr.StreamingRecognizeResponse],
    output_file: Optional[Union[Union[os.PathLike, str, TextIO], List[Union[os.PathLike, str, TextIO]]]] = None,
    additional_info: str = 'no',
    word_time_offsets: bool = False,
    show_intermediate: bool = False,
    file_mode: str = 'w',
):
    global textReturned
     
    if additional_info not in PRINT_STREAMING_ADDITIONAL_INFO_MODES:
        raise ValueError(
            f"Not allowed value '{additional_info}' of parameter `additional_info`. "
            f"Allowed values are {PRINT_STREAMING_ADDITIONAL_INFO_MODES}"
        )
    if additional_info != PRINT_STREAMING_ADDITIONAL_INFO_MODES[0] and show_intermediate:
        warnings.warn(
            f"`show_intermediate=True` will not work if "
            f"`additional_info != {PRINT_STREAMING_ADDITIONAL_INFO_MODES[0]}`. `additional_info={additional_info}`"
        )
    if additional_info != PRINT_STREAMING_ADDITIONAL_INFO_MODES[1] and word_time_offsets:
        warnings.warn(
            f"`word_time_offsets=True` will not work if "
            f"`additional_info != {PRINT_STREAMING_ADDITIONAL_INFO_MODES[1]}`. `additional_info={additional_info}"
        )
    if output_file is None:
        output_file = [sys.stdout]
    elif not isinstance(output_file, list):
        output_file = [output_file]
    file_opened = [False] * len(output_file)
    try:
        for i, elem in enumerate(output_file):
            if isinstance(elem, io.TextIOBase):
                file_opened[i] = False
            else:
                file_opened[i] = True
                output_file[i] = Path(elem).expanduser().open(file_mode)
        start_time = time.time()  # used in 'time` additional_info
        num_chars_printed = 0  # used in 'no' additional_info
        for response in responses:
            if not response.results:
                continue
            partial_transcript = ""
            for result in response.results:
                if not result.alternatives:
                    continue
                transcript = result.alternatives[0].transcript
                if additional_info == 'no':
                    if result.is_final:
                        if show_intermediate:
                            overwrite_chars = ' ' * (num_chars_printed - len(transcript))                           
                            num_chars_printed = 0
                        else:
                            for i, alternative in enumerate(result.alternatives):                                
                                textReturned+=(f'(alternative {i + 1})' if i > 0 else '') + f' {alternative.transcript}'  
                            print('###########'+textReturned)
                            st.markdown(f"**Text:** {textReturned}")
                            with open("OnlineASR_file.txt", "a") as f:
                                f.write(textReturned + "\n")  # Append intermediate transcript to the file                          
                            textReturned=''                            
                    else:
                        partial_transcript += transcript
                elif additional_info == 'time':
                    if result.is_final:
                        for i, alternative in enumerate(result.alternatives):
                            for f in output_file:
                                f.write(
                                    f"Time {time.time() - start_time:.2f}s: Transcript {i}: {alternative.transcript}\n"
                                )
                        if word_time_offsets:
                            for f in output_file:
                                f.write("Timestamps:\n")
                                f.write('{: <40s}{: <16s}{: <16s}\n'.format('Word', 'Start (ms)', 'End (ms)'))
                                for word_info in result.alternatives[0].words:
                                    f.write(
                                        f'{word_info.word: <40s}{word_info.start_time: <16.0f}'
                                        f'{word_info.end_time: <16.0f}\n'
                                    )
                    else:
                        partial_transcript += transcript
                else:  # additional_info == 'confidence'
                    if result.is_final:
                        for f in output_file:
                            f.write(f'## {transcript}\n')
                            f.write(f'Confidence: {result.alternatives[0].confidence:9.4f}\n')
                    else:
                        for f in output_file:
                            f.write(f'>> {transcript}\n')
                            f.write(f'Stability: {result.stability:9.4f}\n')
            if additional_info == 'no':
                if show_intermediate and partial_transcript != '':
                    overwrite_chars = ' ' * (num_chars_printed - len(partial_transcript))
                    for i, f in enumerate(output_file):
                        f.write(">> " + partial_transcript + ('\n' if file_opened[i] else overwrite_chars + '\r'))
                    num_chars_printed = len(partial_transcript) + 3
            elif additional_info == 'time':
                for f in output_file:
                    if partial_transcript:
                        f.write(f">>>Time {time.time():.2f}s: {partial_transcript}\n")
            else:
                for f in output_file:
                    f.write('----\n')
    finally:        
        for fo, elem in zip(file_opened, output_file):
            if fo:
                elem.close()

# ================FILE CONVERSION=================================
def convert_to_wav(uploaded_file,output_file):
    # # Save the uploaded file as recording.wav
    # with open("recording.wav", "wb") as f:
    #     f.write(uploaded_file.read())

    # Convert the audio file using pydub
    audio = AudioSegment.from_file(uploaded_file)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    audio = audio.set_frame_rate(16000)
    audio.export(output_file, format="wav")

    # Read the converted file
    with open("recording_upd.wav", "rb") as f:
        converted_bytes = f.read()

    return converted_bytes

# ==============IMAGE CAPTIONING================================
# Function to process the drawn image and generate captions
# def process_image_and_generate_captions(image, num_captions=1):
#     # Preprocess the image and convert it to a tensor
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = Image.fromarray(image).convert("RGB")
#     image = processor(images=image, return_tensors="pt")

#     # Prepare text captions for CLIP model
#     captions = [
#         "A photo of a dog.",
#         "A beautiful landscape.",
#         "An abstract painting.",
#     ]  # You can specify your own captions here

#     text_inputs = processor(captions, return_tensors="pt", padding=True)

#     # Generate captions using the CLIP model
#     with torch.no_grad():
#         outputs = model(
#             pixel_values=image.pixel_values,
#             attention_mask=image.attention_mask,
#             text_input_ids=text_inputs.input_ids,
#             text_attention_mask=text_inputs.attention_mask,
#             return_dict=True,
#         )

#     # Get the logits for the captions
#     logits_per_image = outputs.logits_per_image

#     # Pick the top num_captions captions for the image
#     _, indices = logits_per_image.topk(num_captions, dim=1)

#     # Decode the indices to get the actual captions
#     captions = [captions[index] for index in indices[0].tolist()]

#     return captions

# ===============STREAMLIT TABS=================================
with st.sidebar:
    tabs = on_hover_tabs(tabName=['Offline ASR', 'Streaming ASR', 'Text To Speech','Drawboard'], 
                         iconName=['play_circle', 'record_voice_over','record_voice_over','play_circle'], default_choice=0)


# ====================MAIN==========================

# ======================OFFLINE ASR=====================
if tabs == 'Offline ASR':
    st.header("Offline ASR Demo")
    st.caption("Ensure the audio file is in WAV format")
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        if not uploaded_file.name.lower().endswith('.wav'):
            file_path = "converted_file.wav"
            with open(file_path, "rb") as file:
                converted_file = file.read()
            
            # convert_to_wav(uploaded_file,converted_file)
             # Save the uploaded file with its original name
            file_name = uploaded_file.name
            file_path = os.path.join(os.getcwd(), file_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1)
            audio = audio.set_sample_width(2)
            audio = audio.set_frame_rate(16000)
            audio.export(converted_file, format="wav")
            uploaded_file=converted_file
    
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.read()

        my_wav_file = uploaded_file.name
        offline_config.sample_rate_hertz = 16000

        # Save the uploaded file as recording.wav
        with open("recording.wav", "wb") as f:
            f.write(bytes_data)

        # Convert the audio file using pydub
        audio = AudioSegment.from_file("recording.wav")
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)
        audio = audio.set_frame_rate(16000)
        audio.export("recording_upd.wav", format="wav")

        # Read the converted file
        with open("recording_upd.wav", "rb") as f:
            converted_bytes = f.read()

        start = time.time()
        with st.spinner('Transcribing...'):
            response = asr_service.offline_recognize(converted_bytes, offline_config)
            asr_best_transcript = ''
            for x in response.results:
                asr_best_transcript += x.alternatives[0].transcript
        timeTaken = time.time() - start
        print(f'Time: {timeTaken}')
        st.markdown("##### ASR Transcript:   \n"
                    + asr_best_transcript)
        
        req["text"] = asr_best_transcript
        resp = riva_tts.synthesize(**req)
        
        obj = wave.open('myaudiofile.wav','wb')
        obj.setnchannels(1) # mono
        obj.setsampwidth(2)
        obj.setframerate(44100)
        obj.writeframes(resp.audio)
        obj.close()
        
        audio_file = open('myaudiofile.wav', 'rb')
        st.markdown("##### ASR Audio:   \n")
        st.audio( audio_file, format="audio/wav")
        st.markdown("##### Listen Again:   \n")
        st.audio(converted_bytes, format="audio/wav")
        
# ===================TEXT TO SPEECH====================
        
elif tabs == 'Text To Speech':
    st.header("Text To Speech Demo")
    st.caption("Please Enter some input before using any of the buttons.")
    input_text = st.text_area(
        label="Enter text to convert...",
        value="",
        height=200
    )
    
    if input_text:
        result = gf.correct(input_text, max_candidates=1)
        result=list(result)
        print(result[0])
        corrected_text = result[0]
        st.text('Corrected Text: '+corrected_text)
    
    # Add a checkbox for the user to choose whether to use the corrected text
    st.caption("Select the checkbox below if you feel the corrected text above is better.")
    use_corrected_text = st.checkbox("Use corrected text ", value=False)
    if use_corrected_text:
        input_text = corrected_text
    
    if st.button('Convert to audio'):
        req["text"] = input_text  # Use either original or corrected text for TTS
        resp = riva_tts.synthesize(**req)
        
        obj = wave.open('myaudiofile.wav','wb')
        obj.setnchannels(1) # mono
        obj.setsampwidth(2)
        obj.setframerate(44100)
        obj.writeframes(resp.audio)
        obj.close()
        
        audio_file = open('myaudiofile.wav', 'rb')
        st.audio(audio_file, format="audio/wav")
    
    if st.button("Get Suggestions"):
        try:
            print("Next Word Prediction with Pytorch using BERT")
            no_words_to_be_predicted, select_model, enter_input_text = set_model_config(no_words_to_be_predicted=5, select_model = "bert", enter_input_text = input_text)
            if select_model.lower() == "bert":
                bert_tokenizer, bert_model  = load_model(select_model)
                res = get_prediction_end_of_sentence(enter_input_text, select_model)
                answer_bert = []
                print(res['bert'].split("\n"))
                for i in res['bert'].split("\n"):
                    answer_bert.append(i)
                    answer_as_string_bert = "    ".join(answer_bert)
        except Exception as e:
            print('Some problem occured')
            
        suggestions=answer_bert

        # Display the suggestions
        st.text("Next word suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"{i}. {suggestion}")

# ====================STREAMING ASR=======================

elif tabs == 'Streaming ASR':
    st.header("Streaming ASR Demo")
    
       
    riva.client.audio_io.list_input_devices()
    default_device_info = riva.client.audio_io.get_default_input_device_info()
    print(default_device_info)
    default_device_index = None if default_device_info is None else default_device_info['index']
    responses=[]
   # default_device_index = 7
    #input_device = 7  # default device
    #Please change this index if the mic is not recognized properly. 
    input_device = default_device_index
    streaming_config.config.sample_rate_hertz = 16000
    
    col1, col2 = st.columns(2)
    with col1:
        start_execution = st.button('Start Streaming')
    with col2:
        stop_exec = st.button('Stop Streaming')
    
    if start_execution:        
        file_ = open("sound.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" width="100" height="100" alt="sound gif">',
            unsafe_allow_html=True,
        )
                
        micInput = riva.client.audio_io.MicrophoneStream(
            rate=16000,
            chunk=16000,
            device=input_device,
        )
        with micInput as audio_chunk_iterator:     
            if stop_exec:
                micInput.close()   
            print_streaming(
                responses=asr_service.streaming_response_generator(
                    audio_chunks=audio_chunk_iterator,
                    streaming_config=streaming_config,
                ),
                show_intermediate=False,
                additional_info = 'no',
            )

    # Display the concatenated paragraph
    if stop_exec:
        st.markdown("### Transcription:")
        final_transcription = ""
        with open("OnlineASR_file.txt", "r") as f:
            final_transcription = f.read()  # Read the contents of the file

        st.markdown(final_transcription)  # Display the final transcript
        
        req["text"] = final_transcription
        resp = riva_tts.synthesize(**req)
        
        obj = wave.open('myaudiofile.wav','wb')
        obj.setnchannels(1) # mono
        obj.setsampwidth(2)
        obj.setframerate(44100)
        obj.writeframes(resp.audio)
        obj.close()
        
        audio_file = open('myaudiofile.wav', 'rb')
        st.markdown("##### ASR Audio:   \n")
        st.audio( audio_file, format="audio/wav")
        
        # Clear the file to prepare it for the next streaming session
        with open("OnlineASR_file.txt", "w") as f:
            f.write("")  # Clear the contents of the file
 
# ==================DRAWBOARD=============================
    
elif tabs == 'Drawboard':
    st.header("Drawboard Demo")
    st.write("Draw your mind if you can't put it into words")
    # Create a canvas for drawing
    canvas_result= st_canvas(                                           #canvas_result = 
        fill_color="#59a697",
        stroke_width=10,
        stroke_color="#000000",
        background_color="#59a697",
        width=800,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("Save"):
        if canvas_result.image_data is not None:
            imageio.imwrite("drawboard\drawn_image.png", canvas_result.image_data)
            
    
    # # Generate captions when the "Get Caption" button is clicked
    # if st.button("Get Caption"):
    #     if canvas_result.image_data is not None:
    #         image = canvas_result.image_data.astype("uint8")
    #         num_captions = 3  # You can specify the number of captions you want
    #         captions = process_image_and_generate_captions(image, num_captions)
    #         st.subheader("Generated Captions:")
    #         for i, caption in enumerate(captions, 1):
    #             st.write(f"{i}. {caption}")
    #     else:
    #         st.warning("Please draw a picture first.")
    
    

