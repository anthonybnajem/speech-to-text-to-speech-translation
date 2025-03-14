import logging
import socket
import time
import threading
from datetime import datetime, timezone
import speech_recognition as sr
import numpy as np
import sounddevice as sd
import pyttsx3  # Added for speech output
from utils.print_audio import print_sound, get_volume_norm, convert_and_normalize

class AudioSocketClient:
    CHANNELS = 1
    RATE = 16000
    CHUNK = 4096
    PHRASE_TIME_LIMIT = 2
    PAUSE_THRESHOLD = 0.8
    RECORDER_ENERGY_THRESHOLD = 1000
    MAX_SPEECH_ATTEMPTS = 3  # Maximum times to retry speech recognition

    def __init__(self) -> None:
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.RECORDER_ENERGY_THRESHOLD
        self.recorder.dynamic_energy_threshold = False
        self.recorder.pause_threshold = self.PAUSE_THRESHOLD

        self.tts_engine = pyttsx3.init()  # Initialize TTS engine
        self.tts_engine.setProperty("rate", 170)  # Set voice speed
        self.list_audio_devices()
        self.input_device_index, self.output_device_index = sd.default.device
    
        
        # self.speak(f"Using input index {self.input_device_index} and output index {self.output_device_index}.")

        self.confirm_and_set_device_indices()

        self.source = sr.Microphone(device_index=self.input_device_index, sample_rate=self.RATE)
        self.transcription = [""]
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.time_last_sent = None
        self.time_first_received = None
        self.time_last_received = None
        self.volume_input = 0
        self.volume_output = 0
        self.time_flush_received = 2
        
        threading.Thread(target=self.__debug_worker__, daemon=True).start()
    
    def list_audio_devices(self):
        """Lists all available input and output devices and allows user selection."""
        devices = sd.query_devices()
        self.input_devices = []
        self.output_devices = []

        print("Available audio devices:\n")
        self.speak("Here are the available audio devices.")
        
        for i, device in enumerate(devices):
            channels_in, channels_out = device['max_input_channels'], device['max_output_channels']
            device_name = device['name']
            if channels_in > 0:
                self.input_devices.append((i, device_name))
                self.speak(f"Input Microphone [{i}]: {device_name}")
            if channels_out > 0:
                self.output_devices.append((i, device_name))
                self.speak(f"Output Speakers [{i}]: {device_name}")

        if not self.input_devices or not self.output_devices:
            self.speak("No valid input or output devices found. Please check your system.")

        self.input_device_index = self.input_devices[0][0] if self.input_devices else None
        self.output_device_index = self.output_devices[0][0] if self.output_devices else None

        self.speak(f"Using input device {self.input_device_index} and output device {self.output_device_index}.")

    
    def speak(self, text):
        """Speaks out the given text using TTS."""
        print(text)
        self.tts_engine.endLoop() if self.tts_engine._inLoop else None  # Fix TTS loop error
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def confirm_and_set_device_indices(self):
        self.speak("Is this correct? Say yes or no, or type new indices.")

        response = self.get_speech_or_text_input()

        if response in ["yes", "y"]:
            return
        elif response in ["no", "n"]:
            self.speak("Select new device indices. Say or type the new values.")

        self.input_device_index = self.get_device_index("Enter the index of the physical microphone.")
        self.output_device_index = self.get_device_index("Enter the index of the output speaker.")

    def get_speech_or_text_input(self):
        """Tries speech input multiple times before falling back to manual input."""
        attempts = 0
        while attempts < self.MAX_SPEECH_ATTEMPTS:
            response = self.get_speech_input()
            if response:
                return response
            attempts += 1
            self.speak("Could not understand audio. Please try again.")

        self.speak("Could not understand after multiple attempts. Please type your response.")
        return input("> ").strip().lower()

    def get_speech_input(self):
        """Attempts to get user input via speech. Returns recognized text or None if unsuccessful."""
        try:
            with sr.Microphone() as source:
                self.recorder.adjust_for_ambient_noise(source)
                self.speak("Listening for response. Speak now.")
                audio = self.recorder.listen(source, timeout=5)
                response = self.recorder.recognize_google(audio).lower()
                self.speak(f"You said: {response}")
                return response
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            return None  

    def get_device_index(self, prompt):
        """Tries speech input first, then falls back to manual input."""
        self.speak(prompt)
        attempts = 0

        while attempts < self.MAX_SPEECH_ATTEMPTS:
            response = self.get_speech_input()
            if response:
                response = self.convert_number_words_to_digits(response)
                if response.isdigit():
                    return int(response)
                self.speak("Invalid input. Please try again.")
            attempts += 1

        self.speak("Could not understand. Please type the index.")
        return int(input("> ").strip())

    def convert_number_words_to_digits(self, text):
        """Converts spoken number words into actual digits."""
        number_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
        }
        return " ".join([number_map.get(word, word) for word in text.split()])

    def record_callback(self, _, audio: sr.AudioData):
        data = audio.get_raw_data()
        self.time_last_sent = time.time()
        logging.debug("send audio data %f", self.time_last_sent)
        self.socket.send(data)
        self.volume_input = get_volume_norm(convert_and_normalize(np.frombuffer(data, dtype=np.int16)))
    
    def start(self, ip, port):
        self.speak(f"Attempting to connect to IP {ip}, port {port}")
        self.socket.connect((ip, port))
        self.speak(f"Successfully connected to IP {ip}, port {port}")

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
        
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=None)
        self.speak("Listening now.")

        self.volume_print_worker = threading.Thread(target=self.__volume_print_worker__, daemon=True)
        self.volume_print_worker.start()
        
        with sd.OutputStream(samplerate=self.RATE, channels=1, dtype=np.float32, device=self.output_device_index) as audio_output:
            try:
                while True:
                    packet = self.socket.recv(self.CHUNK)
                    if packet:
                        audio_chunk = np.frombuffer(packet, dtype=np.float32)
                        self.time_last_received = time.time()
                        if not self.time_first_received:
                            logging.debug("First audio packet - time: %f", self.time_last_received - self.time_last_sent)
                            self.time_first_received = self.time_last_received
                        audio_output.write(audio_chunk)
                        self.volume_output = get_volume_norm(audio_chunk)
            except ConnectionResetError:
                self.speak("Server connection reset. Shutting down client.")
            except KeyboardInterrupt:
                self.speak("Received keyboard input. Shutting down.")
        
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
    
    def __volume_print_worker__(self):
        last_volume_input = 0
        last_volume_output = 0
        print_sound(0, 0, blocks=10)
        while True:
            if abs(last_volume_input - self.volume_input) > 0.1 or abs(last_volume_output - self.volume_output) > 0.1:
                print_sound(self.volume_input, self.volume_output, blocks=10)
                last_volume_input = self.volume_input
                last_volume_output = self.volume_output
            if self.time_last_sent and time.time() - self.time_last_sent > self.PHRASE_TIME_LIMIT:
                self.volume_input = 0
            time.sleep(0.1)
    
    def __debug_worker__(self):
        self.speak("Started background debug worker.")
        while True:
            time.sleep(1)

if __name__ == "__main__":
    date_str = datetime.now(timezone.utc)
    logging.basicConfig(filename=f"logs/{date_str}-output.log", encoding='utf-8', level=logging.DEBUG)
    print('\033[?25l', end="")
    client = AudioSocketClient()
    client.start('localhost', 4444)
    print('\033[?25h', end="")
