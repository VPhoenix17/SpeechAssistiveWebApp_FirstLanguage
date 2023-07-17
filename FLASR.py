import io
import os
import base64
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import (Iterable, List, Optional,
                    TextIO, Union)
import riva.client
import riva.client.audio_io
import riva.client.proto.riva_asr_pb2 as rasr
import streamlit as st
import wave
from pydub import AudioSegment
# import streamlit_ace as st_ace

# Grammar imports
from happytransformer import HappyTextToText, TTSettings
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=1, min_length=1)

#suggestion generator imports
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



from st_on_hover_tabs import on_hover_tabs


#DO NOT CHANGE THIS VALUE
uri = "ec2-54-236-5-236.compute-1.amazonaws.com:50051"

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

#text completion suggestions generator
def generate_completions(input_text, max_length=20, num_completions=5):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        num_return_sequences=num_completions,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        early_stopping=True
    )
    completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
    return completions



PRINT_STREAMING_ADDITIONAL_INFO_MODES = ['no', 'time', 'confidence']
textReturned = ''



#print streaming
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


with st.sidebar:
    tabs = on_hover_tabs(tabName=['Offline ASR', 'Streaming ASR', 'Text To Speech'], 
                         iconName=['play_circle', 'record_voice_over','record_voice_over'], default_choice=0)

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
        
    
        
elif tabs == 'Text To Speech':
    input_text = st.text_area(
        label="Enter text to convert...",
        value="",
        height=200
    )
    
    # Perform grammar correction using the happy_tt.generate_text() function
    result = happy_tt.generate_text(input_text, args=args)
    corrected_text = result.text
    st.text('Corrected Text: '+corrected_text)
    
    # Add a checkbox for the user to choose whether to use the corrected text
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
        completions = generate_completions(input_text)
        if completions:
            st.markdown('### Suggestions:')
            for completion in completions:
                st.markdown(f'- {completion}')


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
    
       



