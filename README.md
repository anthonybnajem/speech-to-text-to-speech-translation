# Speech-To-Text-To-Speech Translation

This project enables real-time audio-to-audio translation over sockets using OpenAI's Whisper for speech-to-text and Microsoft SpeechT5 for text-to-speech synthesis. The system consists of a **client-server** setup, where audio is sent from the client to the server, transcribed into English, converted back into speech, and streamed back to the client.

## **Features**
- **Local real-time speech translation**
- **Uses OpenAI Whisper for speech recognition**
- **SpeechT5 for text-to-speech synthesis**
- **Client-server architecture over WebSockets**
- **Supports virtual audio routing for integration with video calls**

## **1. Install Prerequisites**

### **Update macOS and Developer Tools**
Ensure **Xcode Command Line Tools** are installed and updated:
```bash
xcode-select --install
softwareupdate --all --install --agree-to-license
```

### **Install Homebrew (if not installed)**
Check if Homebrew is installed:
```bash
brew --version
```
If not installed, run:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### **Install FFmpeg**
```bash
brew install ffmpeg
```

---


## **. Alternative Setup Without Anaconda**

You can use **Python's built-in `venv`** instead of Anaconda:

### **1️⃣ Install Python 3.11**
Check your Python version:
```bash
python3 --version
```
If it’s not **3.11 or later**, install it via Homebrew:
```bash
brew install python@3.11
```

### **2️⃣ Create a Python Virtual Environment**
```bash
python3.11 -m venv speech-to-speech-env
source speech-to-speech-env/bin/activate
```


## **2. Install Anaconda and Set Up Environment**

### **Download and Install Anaconda**
Download **Anaconda** from: [https://www.anaconda.com/download](https://www.anaconda.com/download)

### **Initialize Conda**
```bash
conda init
```
Restart your terminal or run:
```bash
source ~/.zshrc  # If using zsh
source ~/.bashrc  # If using bash
```

### **Update Conda**
```bash
conda upgrade conda
```

---

## **3. Create Virtual Environment and Install Dependencies**

### **Create a Conda Virtual Environment**
```bash
conda create -n speech-to-speech python=3.11 -y
```
Activate the environment:
```bash
conda activate speech-to-speech
```

### **Install PyTorch**
Visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Install PyTorch with the appropriate command for your system:
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

### **Install Required Libraries**
```bash
conda install -c conda-forge librosa
conda install -c huggingface transformers
pip install pyaudio
```

---

## **4. Setup and Run the Server**
Navigate to the **server directory** and install dependencies:
```bash
cd server
pip install -r requirements.txt
```
Run the server:
```bash
python server.py
```

---

## **5. Setup and Run the Client**

### **(Optional) Install Virtual Audio Device for Routing Audio**
- Install **BlackHole**: [https://existential.audio/blackhole/](https://existential.audio/blackhole/)
- Follow their setup guide to configure it as a **virtual microphone**.

### **Install Client Dependencies**
Navigate to the **client directory**:
```bash
cd ../client
pip install -r requirements.txt
```

### **Configure Client for Remote Server (If Needed)**
If using a **remote server**, update `client.py`:
```python
client.start(("your-server-ip", 4444))
```
For **local testing**, keep `localhost`.

### **Run the Client**
```bash
python client.py
```

---

## **6. Troubleshooting**

### **Common Errors and Fixes**

#### **Missing `libpython3.11.a` or Clang Errors**
```bash
clang: error: no such file or directory: '/Users/youruser/anaconda3/envs/speech-to-speech/lib/python3.11/config-3.11-darwin/libpython3.11.a'
```
✅ **Solution:** Ensure Conda and dependencies are updated:
```bash
conda update conda
brew update && brew upgrade
```
Recreate the environment:
```bash
conda deactivate
conda env remove -n speech-to-speech
conda create -n speech-to-speech python=3.11 -y
```

#### **`PY_SSIZE_T_CLEAN macro must be defined` Error**
```bash
conda deactivate
conda activate speech-to-speech
pip install --upgrade pip setuptools wheel
```

---


### **3️⃣ Install Dependencies**
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install librosa transformers pyaudio websockets sounddevice
pip install -r requirements.txt
```

### **4️⃣ Run the Server and Client**
```bash
python server.py  # Start server
python client.py  # Start client
```

---

## **8. Fix PyAudio Installation on macOS (Apple Silicon)**

### **1️⃣ Install PortAudio**
```bash
brew install portaudio
```

### **2️⃣ Set Environment Variables**
```bash
export LDFLAGS="-L$(brew --prefix portaudio)/lib"
export CFLAGS="-I$(brew --prefix portaudio)/include"
```

### **3️⃣ Install PyAudio**
```bash
pip install pyaudio
```
If the above fails, try:
```bash
pip install --no-cache-dir pyaudio
```

### **4️⃣ Retry Setup**
```bash
pip install -r requirements.txt
python server.py
python client.py
```

---

## **🎤🎧 Everything is set! You should now be able to transcribe and translate speech in real-time. 🚀**

---

Let me know if you need further assistance! 🚀💡




### Data flow
![Client Server Data Flow](https://github.com/kensonhui/live-speech-to-text-to-speech/assets/60726802/6a81c04e-c493-43d0-ad2e-a61638ddb81b)

### Server-side Flow
![Server Flow Diagram](https://github.com/kensonhui/live-speech-to-text-to-speech/assets/60726802/87ba0b85-6c7a-4cb6-bf19-f2fdf3722455)

Within the client, the user can pipe in the audio output to any virtual microphone or audio device they would like. One application is for video calls, the user can pipe the output to a virtual microphone, then use that audio device in a meeting so that everything they say is translated.

### Server Installation Instructions:
#### These are all important steps!

Ensure your ports specified in server.py is open! The default port we chose was 4444.

Make sure your XCode CLI or C++ compiler tools are fully updated!
https://developer.apple.com/forums/thread/677124

Install FFmpeg:
```sudo apt install ffmpeg```

Install Anaconda for your device:
https://www.anaconda.com/download

Run the initialization command:
```conda init```

Make sure Conda is updated to the latest version:
```conda upgrade conda```

Create a virtual environment
```conda create -n "speech-to-speech" python==3.11```

```conda activate speech-to-speech```

Install Pytorch here:
https://pytorch.org/get-started/locally/

Install Librosa
```conda install -c conda-forge librosa```

Install Transformers:
```conda install -c huggingface transformers```

Install pyaudio:
```pip install pyaudio```

Install Project Packages:
```cd server```
```pip install -r requirements.txt```

Finally run the server:
```python server.py```


### Client Installation
If you'd like to use the translation in a video call or such, you can install software to create a virtual microphone. On Mac you can use Blackhole.

Install FFmpeg - you can do so with brew, or here: https://ffmpeg.org/download.html.
```brew install ffmpeg``` 

Install requirements.txt in the clients folder
```pip install -r requirements.txt```

If you're running server.py on a remote server, change "localhost" to your remote server ip in client .py in this line:
```
client.start(("localhost", 4444)) 
```

Finally run the client:
```python client.py```

Within the client, you can select the appropriate input and output device that audio will be piped through.

### Notebooks for Testing

## speech.ipynb

Audiofile -> Translates to english text with whisper -> Create a synthesized voice with MS T5 TTS

## transcribe.ipynb

Microphone -> Transcribe to english text

## speech-to-transcribe

Microphone -> Translate and transcribe to english text


### Errors
```clang: error: no such file or directory: '/Users/kensonhui/anaconda3/envs/speech-to-speech/lib/python3.11/config-3.11-darwin/libpython3.11.a'```

or 

```PY_SSIZE_T_CLEAN macro must be defined for '#' formats```

You'll have to update conda, update XCode, update brew, update your XCode CLI tools. Destroy your env, and rebuild your environment :D.

