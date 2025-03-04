""" Server for real-time translation and voice synthesization """
from typing import Dict
from queue import Queue
import select
import socket
import pyaudio
import torch
from models.speech_recognition import SpeechRecognitionModel
from models.text_to_speech import TextToSpeechModel

class AudioSocketServer:
    """ Class that handles real-time translation and voice synthesization
        Socket input -> SpeechRecognition -> text -> TextToSpeech -> Socket output
    """
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096
    PORT = 4444
    BACKLOG = 5  # Max number of pending connections

    def __init__(self, whisper_model):
        print("[🔧] Initializing server...")
        
        self.audio = pyaudio.PyAudio()
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.data_queue: Queue = Queue()

        # Initialize the speech recognition model (Whisper)
        print("[🎙️] Loading Whisper speech recognition model...")
        self.transcriber = SpeechRecognitionModel(
            model_name=whisper_model,
            data_queue=self.data_queue,
            generation_callback=self.handle_generation,
            final_callback=self.handle_transcription
        )

        # Initialize text-to-speech model
        print("[🔊] Loading Text-to-Speech model...")
        self.text_to_speech = TextToSpeechModel(callback_function=self.handle_synthesize)
        self.text_to_speech.load_speaker_embeddings()

        self.read_list = []

        print("[✅] Server initialized successfully!")

    def __del__(self):
        """ Cleanup on object deletion """
        print("[⚠️] Shutting down server...")
        self.cleanup()

    def handle_generation(self, packet: Dict):
        """ Placeholder function for transcription """
        print(f"[📝] Partial transcription received: {packet}")

    def handle_transcription(self, packet: str, client_socket):
        """ Finalized transcription callback """

        if packet.strip():
            print(f"[✅ Detected Speech]: {packet}")
        else:
            print("[⚠️] Received empty transcription!")

        # Send the transcription to the TTS model
        print("[🔄] Sending text for synthesis...")
        self.text_to_speech.synthesise(packet, client_socket)

    def handle_synthesize(self, audio: torch.Tensor, client_socket):
        """ Callback function to stream audio back to the client """
        if client_socket is None:
            print("[⚠️] Warning: Client socket is None. Skipping audio streaming.")
            return
        print(f"[🔊] Streaming synthesized audio to {client_socket.getpeername()}")
        self.stream_numpy_array_audio(audio, client_socket)

    def start(self):
        """ Starts the server """
        self.transcriber.start(16000, 2)
        
        print(f"[🎧] Listening for connections on port {self.PORT}...")
        self.serversocket.bind(('', self.PORT))
        self.serversocket.listen(self.BACKLOG)
        
        self.read_list = [self.serversocket]

        try:
            while True:
                readable, _, _ = select.select(self.read_list, [], [])
                for s in readable:
                    if s is self.serversocket:
                        (clientsocket, address) = self.serversocket.accept()
                        self.read_list.append(clientsocket)
                        print(f"[🟢] New connection from {address}")
                    else:
                        try:
                            data = s.recv(4096)
                            if data:
                                print(f"[🎙️] Received {len(data)} bytes of audio data from client.")
                                self.data_queue.put((s, data))
                            else:
                                print(f"[🔴] Client disconnected: {s.getpeername()}")
                                self.read_list.remove(s)
                        except ConnectionResetError:
                            print(f"[⚠️] Connection lost with client {s.getpeername()}")
                            self.read_list.remove(s)
        except KeyboardInterrupt:
            print("[🛑] Server stopping...")

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """ Graceful server cleanup """
        print("[🧹] Performing cleanup...")
        try:
            self.audio.terminate()
            self.transcriber.stop()
            self.serversocket.shutdown(socket.SHUT_RDWR)
            self.serversocket.close()
            print("[✅] Server shutdown complete.")
        except OSError as e:
            print(f"[⚠️] OSError during cleanup: {e}")

    def stream_numpy_array_audio(self, audio, client_socket):
        """ Streams audio back to the client"""
        if client_socket is None:
            print("[⚠️] Warning: Attempted to send audio to a NoneType client. Skipping.")
            return
        try:
            client_socket.sendall(audio.numpy().tobytes())
        except ConnectionResetError as e:
            print(f"[❌] Error sending audio to client: {e}")
            if client_socket in self.read_list:
                self.read_list.remove(client_socket)
        except AttributeError:
            print("[⚠️] Client socket became None during transmission. Skipping audio.")

if __name__ == "__main__":
    print("[🚀] Starting Speech-to-Speech Translation Server...")
    server = AudioSocketServer(whisper_model="base")
    server.start()
