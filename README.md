# ZhenGPT Framework

## Setup

### Setting Up Quantization
```
pip3 install --upgrade transformers optimum
```
**If using PyTorch 2.1 + CUDA 12.x:**
```
pip3 install --upgrade auto-gptq
```
**or, if using PyTorch 2.1 + CUDA 11.x:**
```
pip3 install --upgrade auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```

### Voice Recognition
```
pip install pocketsphinx
```

### PyAudio

**Windows**
```
pip install pyaudio
```

**Linux**
```
pip install pyaudio
```
or
```
sudo apt-get install python-pyaudio python3-pyaudio
```
or
```
sudo apt-get install portaudio19-dev python-all-dev python3-all-dev && sudo pip install pyaudio
```

**OS X**
On OS X, install PortAudio using Homebrew: brew install portaudio. Then, install PyAudio using Pip: pip install pyaudio.
