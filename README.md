# ZhenGPT Framework

## Setup
pip3 install --upgrade transformers optimum
# If using PyTorch 2.1 + CUDA 12.x:
pip3 install --upgrade auto-gptq
# or, if using PyTorch 2.1 + CUDA 11.x:
pip3 install --upgrade auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

PyAudio (for microphone users)
PyAudio is required if and only if you want to use microphone input (Microphone). PyAudio version 0.2.11+ is required, as earlier versions have known memory management bugs when recording from microphones in certain situations.

If not installed, everything in the library will still work, except attempting to instantiate a Microphone object will raise an AttributeError.

The installation instructions on the PyAudio website are quite good - for convenience, they are summarized below:

On Windows, install PyAudio using Pip: execute pip install pyaudio in a terminal.

On Debian-derived Linux distributions (like Ubuntu and Mint), install PyAudio using APT: execute sudo apt-get install python-pyaudio python3-pyaudio in a terminal.
If the version in the repositories is too old, install the latest release using Pip: execute sudo apt-get install portaudio19-dev python-all-dev python3-all-dev && sudo pip install pyaudio (replace pip with pip3 if using Python 3).

On OS X, install PortAudio using Homebrew: brew install portaudio. Then, install PyAudio using Pip: pip install pyaudio.

On other POSIX-based systems, install the portaudio19-dev and python-all-dev (or python3-all-dev if using Python 3) packages (or their closest equivalents) using a package manager of your choice, and then install PyAudio using Pip: pip install pyaudio (replace pip with pip3 if using Python 3).

PyAudio wheel packages for common 64-bit Python versions on Windows and Linux are included for convenience, under the third-party/ directory in the repository root. To install, simply run pip install wheel followed by pip install ./third-party/WHEEL_FILENAME (replace pip with pip3 if using Python 3) in the repository root directory.

pip install pocketsphinx