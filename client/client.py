import logging
import socket
import time
import threading
from datetime import datetime, timezone
import speech_recognition as sr
import numpy as np
import sounddevice as sd
import pyttsx3  # For speech output
from utils.print_audio import print_sound, get_volume_norm, convert_and_normalize
import subprocess
import queue  # For simultaneous input

# ====================================================================================
# AudioSocketClient Class
# ====================================================================================

class AudioSocketClient:
    """
    AudioSocketClient handles audio capture, speech recognition, text-to-speech,
    and socket communication for a speech-to-text-to-speech translation client.
    
    This version merges all available devices (physical input, physical output, and
    Bluetooth devices) into one unified list (self.all_devices) that is printed and used for selection.
    
    A unified function, get_input, is used for all user prompts (via speech or text).
    
    Note: Although the merged list may include Bluetooth devices (e.g. index 5), 
    SpeechRecognition typically sees only physical microphones. If a Bluetooth device 
    is chosen for input and cannot be used for STT, the code will ask you to select a valid 
    physical microphone.
    """

    CHANNELS = 1
    RATE = 16000
    CHUNK = 4096
    PHRASE_TIME_LIMIT = 2
    PAUSE_THRESHOLD = 0.8
    RECORDER_ENERGY_THRESHOLD = 1000
    MAX_SPEECH_ATTEMPTS = 3

    def __init__(self) -> None:
        """
        Initializes the client by setting up the recognizer, TTS engine,
        listing and merging audio devices, and starting background threads.
        """
        # Initialize the speech recognizer.
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.RECORDER_ENERGY_THRESHOLD
        self.recorder.dynamic_energy_threshold = False
        self.recorder.pause_threshold = self.PAUSE_THRESHOLD

        # Initialize the TTS engine.
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 170)  # Initial speech rate

        # List all available devices (physical and Bluetooth) and merge them.
        self.list_audio_devices()
        # For initial selection, use SoundDevice's default device.
        self.input_device_index, self.output_device_index = sd.default.device

        # Confirm or change device indices using the merged list.
        self.confirm_and_set_device_indices()

        # Try to create the microphone using the chosen input index.
        try:
            self.source = sr.Microphone(device_index=self.input_device_index, sample_rate=self.RATE)
        except AssertionError as e:
            self.speak("The selected input device is not available for speech recognition.")
            valid_indices = [i for i, _ in self.input_devices]
            self.input_device_index = self.get_input("Enter a valid index for the microphone device:", numeric=True, valid_indices=valid_indices)
            self.source = sr.Microphone(device_index=self.input_device_index, sample_rate=self.RATE)
        self.transcription = [""]

        # Create a socket for audio streaming.
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.time_last_sent = None
        self.time_first_received = None
        self.time_last_received = None
        self.volume_input = 0
        self.volume_output = 0
        self.time_flush_received = 2

        # Start a background debug worker thread.
        threading.Thread(target=self.__debug_worker__, daemon=True).start()


    def list_audio_devices(self):
        """
        Lists all available audio devices from SoundDevice and merges them with
        additional Bluetooth devices (detected via bluetoothctl). The merged list is
        stored in self.all_devices and announced via TTS.
        """
        devices = sd.query_devices()
        self.input_devices = []
        self.output_devices = []
        self.bluetooth_devices = []

        print("Available audio devices:\n", flush=True)
        self.speak("Here are the available audio devices.")

        for i, device in enumerate(devices):
            channels_in = device.get('max_input_channels', 0)
            channels_out = device.get('max_output_channels', 0)
            device_name = device.get('name', "Unknown Device")
            
            if channels_in > 0:
                self.input_devices.append((i, device_name))
                self.speak(f"Input Microphone [{i}]: {device_name}")
            if channels_out > 0:
                self.output_devices.append((i, device_name))
                self.speak(f"Output Speakers [{i}]: {device_name}")
            if "bluetooth" in device_name.lower() or "airpods" in device_name.lower():
                # For devices from sd.query_devices(), MAC info may be "N/A".
                self.bluetooth_devices.append((i, device_name, "N/A"))
                self.speak(f"Bluetooth Device [{i}]: {device_name}")

        # Detect additional Bluetooth devices.
        self.detect_bluetooth_devices()

        # Merge all devices into one list.
        self.all_devices = []
        for dev in self.input_devices:
            self.all_devices.append((dev[0], dev[1], "physical input"))
        for dev in self.output_devices:
            if dev[0] not in [d[0] for d in self.all_devices]:
                self.all_devices.append((dev[0], dev[1], "physical output"))
        for dev in self.bluetooth_devices:
            if dev[0] not in [d[0] for d in self.all_devices]:
                self.all_devices.append((dev[0], dev[1], "bluetooth"))
        self.all_devices.sort(key=lambda x: x[0])

        # Announce merged list.
        self.speak("Merged list of all available devices:")
        for dev in self.all_devices:
            self.speak(f"Device [{dev[0]}]: {dev[1]} ({dev[2]})")

        if not self.input_devices or not self.output_devices:
            self.speak("No valid input or output devices found. Please check your system.")

        # Set default device indices using physical devices.
        self.input_device_index = self.input_devices[0][0] if self.input_devices else None
        self.output_device_index = self.output_devices[0][0] if self.output_devices else None

        self.speak(f"Using input device {self.input_device_index} and output device {self.output_device_index}.")

        print("Input devices:", self.input_devices, flush=True)
        print("Output devices:", self.output_devices, flush=True)
        print("Bluetooth devices:", self.bluetooth_devices, flush=True)
        print("Merged devices:", self.all_devices, flush=True)


    def detect_bluetooth_devices(self):
        """
        Detects additional Bluetooth audio devices using 'bluetoothctl devices'.
        Each device is stored as a tuple: (index, device_name, mac_address).
        """
        try:
            output = subprocess.check_output("bluetoothctl devices", shell=True).decode()
            devices = [line.split("Device ")[1] for line in output.strip().split("\n") if "Device" in line]
            if devices:
                for i, device in enumerate(devices, start=len(self.input_devices) + len(self.output_devices)):
                    try:
                        device_mac, device_name = device.split(maxsplit=1)
                    except ValueError:
                        device_mac = device
                        device_name = "Unknown Bluetooth Device"
                    if not any(dev[0] == i for dev in self.bluetooth_devices):
                        self.bluetooth_devices.append((i, device_name, device_mac))
                        self.speak(f"Bluetooth Audio Device [{i}]: {device_name} ({device_mac})")
        except Exception as e:
            self.speak(f"Error detecting Bluetooth devices: {e}")


    def connect_bluetooth_device(self):
        """
        Connects to a selected Bluetooth device using 'bluetoothctl connect'.
        (Retained for manual connection; not used in this merged version.)
        """
        if not self.bluetooth_devices:
            self.speak("No Bluetooth devices available to connect.")
            return False

        if len(self.bluetooth_devices) > 1:
            self.speak("Multiple Bluetooth devices found. Please select one by its index.")
            valid_bt_indices = sorted(list({dev[0] for dev in self.bluetooth_devices}))
            selected_index = self.get_input("Enter the Bluetooth device index.", numeric=True, valid_indices=valid_bt_indices)
        else:
            selected_index = self.bluetooth_devices[0][0]

        selected_device = None
        for dev in self.bluetooth_devices:
            if dev[0] == selected_index:
                selected_device = dev
                break

        if selected_device is None:
            self.speak("Selected Bluetooth device not found.")
            return False

        self.speak(f"Attempting to connect to Bluetooth device {selected_device[1]} with MAC {selected_device[2]}.")
        try:
            subprocess.run(f"bluetoothctl connect {selected_device[2]}", shell=True)
            self.speak(f"Connected to Bluetooth device {selected_device[2]}.")
            return True
        except Exception as e:
            self.speak(f"Error connecting to Bluetooth device: {e}")
            return False


    def speak(self, text):
        """
        Speaks the given text using the TTS engine at a faster rate and prints it.
        """
        print(text, flush=True)
        try:
            self.tts_engine.setProperty('rate', 250)
            if self.tts_engine._inLoop:
                self.tts_engine.endLoop()
        except Exception as e:
            print(f"TTS engine error: {e}", flush=True)
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()


    def get_input(self, prompt, valid_options=None, numeric=False, valid_indices=None):
        """
        A unified function to get user input from either speech (STT) or text.
        - If valid_options is provided, the input must match one of them.
        - If numeric is True, the input is converted to an integer and (if valid_indices is provided)
          it must be one of those.
        """
        self.speak(prompt)
        attempts = 0
        while attempts < self.MAX_SPEECH_ATTEMPTS:
            response = self.get_simultaneous_input("Please provide your response:", valid_options)
            response = response.strip().lower()
            if numeric:
                response = self.convert_number_words_to_digits(response)
                if response.isdigit():
                    num = int(response)
                    if valid_indices is None or num in valid_indices:
                        return num
                    else:
                        self.speak(f"Invalid number. Please choose from {valid_indices}.")
                else:
                    self.speak("Invalid input. Please provide a numeric value.")
            else:
                if response:
                    return response
            attempts += 1
        # Fallback to manual input.
        if numeric:
            self.speak(f"Could not understand. Please type a number{' from ' + str(valid_indices) if valid_indices else ''}.")
            while True:
                try:
                    user_input = input("> ").strip()
                    num = int(user_input)
                    if valid_indices is None or num in valid_indices:
                        return num
                    else:
                        print(f"Number must be one of {valid_indices}", flush=True)
                except ValueError:
                    print("Invalid input. Please enter a valid number.", flush=True)
        else:
            self.speak("Could not understand. Please type your response.")
            return input("> ").strip().lower()


    def confirm_and_set_device_indices(self):
        """
        Asks the user to confirm the current device indices or enter new ones.
        The valid indices are taken from the merged device list.
        Note: Although the merged list may include Bluetooth devices, SpeechRecognition
        works only with physical microphones. If a Bluetooth index is chosen for input,
        you will be prompted to select a valid physical microphone.
        """
        self.speak("Is this correct? Say yes or no, or type new indices.")
        response = self.get_input("Please respond with yes or no:", valid_options=["yes", "no", "y", "n"])
        if response in ["yes", "y"]:
            return
        else:
            self.speak("Select new device indices. Please choose from the merged device indexes.")
        valid_indices = [dev[0] for dev in self.all_devices]
        self.input_device_index = self.get_input("Enter the index for the input device:", numeric=True, valid_indices=valid_indices)
        self.output_device_index = self.get_input("Enter the index for the output device:", numeric=True, valid_indices=valid_indices)
        # Try to create the microphone; if invalid, force physical device selection.
        try:
            self.source = sr.Microphone(device_index=self.input_device_index, sample_rate=self.RATE)
        except AssertionError as e:
            self.speak("The selected input device is not available for speech recognition.")
            valid_input_indices = [i for i, _ in self.input_devices]
            self.input_device_index = self.get_input("Enter a valid index for the microphone device:", numeric=True, valid_indices=valid_input_indices)
            self.source = sr.Microphone(device_index=self.input_device_index, sample_rate=self.RATE)


    def get_simultaneous_input(self, prompt, valid_options=None):
        """
        Listens for user input concurrently via speech and text.
        Logs every detected word along with its source.
        If valid_options is provided, repeats until a valid response is received.
        Returns the recognized string.
        """
        while True:
            self.speak(prompt)
            q = queue.Queue()

            def text_worker():
                try:
                    user_input = input("> ").strip().lower()
                    q.put(("text", user_input))
                except Exception as e:
                    logging.debug(f"Text input error: {e}")
                    q.put(("text", ""))

            def speech_worker():
                result = self.get_speech_input()
                if result:
                    q.put(("voice", result))

            t_text = threading.Thread(target=text_worker)
            t_speech = threading.Thread(target=speech_worker)
            t_text.start()
            t_speech.start()

            source, result = q.get()
            print(f"Detected input from {source}: {result}", flush=True)
            if valid_options is None or result in valid_options:
                return result
            else:
                self.speak(f"Invalid input: {result}. Please say " + " or ".join(valid_options))


    def get_speech_input(self):
        """
        Captures audio from the microphone and converts it to text.
        Logs the recognized text as voice-detected.
        """
        try:
            with sr.Microphone() as source:
                self.recorder.adjust_for_ambient_noise(source)
                self.speak("Listening for response. Speak now.")
                audio = self.recorder.listen(source, timeout=5)
                response = self.recorder.recognize_google(audio).lower()
                print(f"Voice detected: {response}", flush=True)
                self.speak(f"You said: {response}")
                return response
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError) as e:
            logging.debug(f"Speech input error: {e}")
            return None  


    def convert_number_words_to_digits(self, text):
        """
        Converts spoken number words into their digit string equivalent.
        Supports numbers from zero up to twenty.
        """
        number_map = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
            "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
            "eighteen": "18", "nineteen": "19", "twenty": "20"
        }
        words = text.split()
        converted_words = [number_map[word] if word in number_map else word for word in words]
        return " ".join(converted_words)


    def record_callback(self, _, audio: sr.AudioData):
        """
        Callback invoked when new audio is captured.
        Sends the raw audio data over the socket and updates the input volume.
        """
        data = audio.get_raw_data()
        self.time_last_sent = time.time()
        logging.debug("Sending audio data at %f", self.time_last_sent)
        try:
            self.socket.send(data)
        except Exception as e:
            logging.debug(f"Error sending audio data: {e}")
        try:
            audio_array = np.frombuffer(data, dtype=np.int16)
            normalized_audio = convert_and_normalize(audio_array)
            self.volume_input = get_volume_norm(normalized_audio)
        except Exception as e:
            logging.debug(f"Error processing audio data: {e}")


    def start(self, ip, port):
        """
        Starts the client by connecting to the given IP and port and starting audio capture.
        (This version does not ask about Bluetooth connection; it simply connects.)
        """
        self.speak(f"Attempting to connect to IP {ip}, port {port}")
        try:
            self.socket.connect((ip, port))
            self.speak(f"Successfully connected to IP {ip}, port {port}")
        except Exception as e:
            self.speak(f"Error connecting to {ip}:{port} - {e}")
            return

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=None)
        self.speak("Listening now.")


    def __volume_print_worker__(self):
        """
        Background worker that prints volume levels of input and output periodically.
        """
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
        """
        Background debug worker that logs activity periodically.
        """
        self.speak("Started background debug worker.")
        while True:
            logging.debug("Debug worker active.")
            time.sleep(1)


# ====================================================================================
# Main Execution
# ====================================================================================

if __name__ == "__main__":
    date_str = datetime.now(timezone.utc)
    logging.basicConfig(filename=f"logs/{date_str}-output.log", encoding='utf-8', level=logging.DEBUG)
    print('\033[?25l', end="")  # Hide cursor
    client = AudioSocketClient()
    client.start('localhost', 4444)
    print('\033[?25h', end="")  # Restore cursor
