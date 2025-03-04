import io
import time
import threading
from queue import Queue
from datetime import datetime, timedelta
import torch
import whisper
import soundfile as sf
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import re
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load a lightweight LLM (change to a more powerful model if needed)
MODEL_NAME = "facebook/opt-1.3b"  # Alternatives: "mistralai/Mistral-7B-Instruct", "facebook/opt-1.3b"



# # Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Load a lightweight LLM (change to a more powerful model if needed)
# MODEL_PATH = "./trained_model"
# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

def only_rewrite_phrase_llm(text):
    """ Use an LLM to paraphrase the phrase in real time """
    prompt = f"Rephrase the below, so that an autistic gamer can understand his teamates, imagine you are translating on the spot for him while playing a game online, give me just the output he should hear --> \"{text}\""
    print(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50, temperature=0.7)

    rewritten_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("rewritten")
    print(rewritten_text.strip())
    print("rewritten end")
    return rewritten_text.replace(prompt, "").strip()  # Remove prompt text

def rewrite_phrase_llm(text):
    """
    1. Load a known dictionary from basic_map.json (which is an array of {"input": "...", "output": "..."} objects).
    2. If any 'input' in that array appears in `text` (case-insensitive),
       replace it and return immediately.
    3. Otherwise, fall back to an LLM paraphrasing approach.
    """

    # Load your replacement mappings from JSON
    with open("basic_map.json", "r", encoding="utf-8") as f:
        rewrite_data = json.load(f)  # rewrite_data is a list of objects: [{"input": "...", "output": "..."}]

    text_lower = text.lower()

    # Check each known phrase for a case-insensitive substring match
    for entry in rewrite_data:
        key = entry["input"]
        replacement = entry["output"]

        # If this known phrase is in our text, do a case-insensitive replacement
        if key.lower() in text_lower:
            # Use a regex to replace exactly the phrase (if you need partial matches, remove \b)
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            replaced_text = pattern.sub(replacement, text)

            return replaced_text  # Return once the first match is handled
        
    return text
    # # -- If no match in the dictionary, fall back to LLM paraphrasing --
    # prompt = (
    #     f"Rephrase the below, so that an autistic gamer can understand his teammates. "
    #     f"Imagine you are translating on the spot for him while playing a game online. "
    #     f"Give me just the output he should hear --> \"{text}\""
    # )
    # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # with torch.no_grad():
    #     output_ids = model.generate(
    #         input_ids,
    #         max_length=50,
    #         temperature=0.7,
    #         # any other generate() parameters you use
    #     )

    # rewritten_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # # You might optionally remove the prompt from the final output
    # # if the model sometimes includes it in the generation:
    # final_text = rewritten_text.replace(prompt, "").strip()

    # return final_text


class SpeechRecognitionModel:
    """ Speech Recognition with Whisper + Real-time Processing """

    def __init__(self, data_queue,
                 generation_callback=lambda *args: None, 
                 final_callback=lambda *args: None, 
                 model_name="base"):

        self.phrase_time = datetime.utcnow()
        self.last_sample = bytes()
        self.data_queue: Queue = data_queue
        self.generation_callback = generation_callback
        self.final_callback = final_callback
        self.phrase_timeout = 1  # Timeout for phrase completion

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model '{model_name}' on {self.device}")

        self.audio_model = whisper.load_model(model_name, device=self.device)
        self.decoding_options: dict = {"task": "translate"}
        print(f"Whisper Decoding Options: {self.decoding_options}")
        
        self.thread = None
        self._kill_thread = False
        self.recent_transcription = ""
        self.current_client = None

    def start(self, sample_rate, sample_width):
        """ Starts the worker thread """
        self.thread = threading.Thread(target=self.__worker__, args=(sample_rate, sample_width))
        self._kill_thread = False
        self.thread.start()

    def stop(self):
        """ Stops the worker thread """
        self._kill_thread = True
        if self.thread:
            self.thread.join()
            self.thread = None

    def __worker__(self, sample_rate, sample_width):
        """ Worker thread event loop """
        while not self._kill_thread:
            now = datetime.utcnow()
            self.__flush_last_phrase__(now)
            if not self.data_queue.empty():
                phrase_complete = self.__update_phrase_time__(now)
                self.__concatenate_new_audio__()
                self.__transcribe_audio__(sample_rate, sample_width, phrase_complete)
            time.sleep(0.05)

    def __update_phrase_time__(self, current_time):
        phrase_complete = False
        if self.phrase_time and current_time - self.phrase_time > timedelta(seconds=self.phrase_timeout):
            self.phrase_time = current_time
            self.last_sample = bytes()
            phrase_complete = True
        return phrase_complete

    def __flush_last_phrase__(self, current_time):
        """ Flush the last phrase if no audio has been sent in a while """
        if self.phrase_time and current_time - self.phrase_time > timedelta(seconds=self.phrase_timeout):
            if self.recent_transcription and self.current_client:
                print(f"Flush {self.recent_transcription}")
                self.final_callback(self.recent_transcription, self.current_client)
                self.recent_transcription = ""
                self.phrase_time = current_time
                self.last_sample = bytes()

    def __concatenate_new_audio__(self):
        while not self.data_queue.empty():
            client, data = self.data_queue.get()
            if client != self.current_client:
                print(f"Flush {self.recent_transcription}")
                self.final_callback(self.recent_transcription, self.current_client)
                self.recent_transcription = ""
                self.phrase_time = datetime.utcnow()
                self.last_sample = bytes()
            self.last_sample += data
            self.current_client = client

    def __transcribe_audio__(self, sample_rate, sample_width, phrase_complete):
        """ Transcribes audio and applies LLM-based phrase rewriting """
        try:
            audio_data = sr.AudioData(self.last_sample, sample_rate, sample_width)
            wav_data = io.BytesIO(audio_data.get_wav_data())
            with sf.SoundFile(wav_data, mode='r') as sound_file:
                audio = sound_file.read(dtype='float32')
                start_time = time.time()

                result = self.audio_model.transcribe(audio, fp16=torch.cuda.is_available(), **self.decoding_options)
                end_time = time.time()

                text = result['text'].strip()
                if text:
                    modified_text = rewrite_phrase_llm(text)  # Rewrite using LLM
                    self.generation_callback({"add": phrase_complete,
                                              "text": modified_text,
                                              "transcribe_time": end_time - start_time})
                    if phrase_complete and self.recent_transcription and self.current_client:
                        print(f"Phrase complete: {self.recent_transcription}")
                        self.final_callback(self.recent_transcription, self.current_client)
                    self.recent_transcription = modified_text
        except Exception as e:
            print(f"Error during transcription: {e}")

    def __del__(self):
        self.stop()

# Custom Callbacks to process real-time transcription with LLM-based phrase rewriting
def custom_generation_callback(packet):
    if "text" in packet:
        modified_text = rewrite_phrase_llm(packet["text"])
        print(f"Modified Transcription: {modified_text}")
        packet["text"] = modified_text

def custom_final_callback(text, client):
    modified_text = rewrite_phrase_llm(text)
    print(f"Final Transcription: {modified_text}")

# Initialize SpeechRecognitionModel with custom callbacks
speech_recognition = SpeechRecognitionModel(
    data_queue=Queue(),
    generation_callback=custom_generation_callback,
    final_callback=custom_final_callback,
    model_name="base"
)
