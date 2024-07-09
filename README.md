
# SRT-to-XTTS-2

This repository features the source code for SRT (SubRip Subtitle) to Speech converter. It is a handy app that turns `.srt` subtitle files into basic `.wav` dubbing/voiceovers using `coqui/XTTS-v2` technology.

### Some potential use cases:

- Making video content more accessible for visually impaired viewers
- Creating dubbed versions of translated subtitles in your own language
- Generating audio versions of subtitled videos for listening on-the-go

## Installation

### Prerequisites

- Windows OS
- Python 3.10 or higher
- CUDA-compatible GPU (recommended for faster processing)

### Requirements
The following environment configuration was used for testing: Windows 10 Pro x64, Python 3.10.11 64-bit, and CUDA 11.8.

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

For CUDA 11.8(GPU):
```bash
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 -f https://download.pytorch.org/whl/torch_stable.html  
```

To install Coqui/XTTS-2 make sure you have git-lfs installed (https://git-lfs.com)
```
git lfs install
```
Install XTTS-2 using the command below. You can install XTTS-2 in a different directory using `cd` command (e.g., `cd c:`), but make sure to specify the path accordingly in `.env` file.
```
git clone https://huggingface.co/coqui/XTTS-v2
```
### Required Environment Variables

Rename the `.env.example` file to `.env` and keep it in the root directory of the project. Adjust the paths below to match your XTTS2 installation directory:

```
# Path to the XTTS2 main directory
XTTS2_PATH="/path/to/your/XTTS-v2"

# Path to the XTTS2 config file
XTTS2_CONFIG="/path/to/your/XTTS-v2/config.json"

# Path to the XTTS2 sample file
XTTS2_SAMPLE="/path/to/your/XTTS-v2/samples/en_sample.wav"
```

## Usage

### 1. Run the main script.
```
python srt2speech.py
```
### 2. Load the `.srt` input file.
A new window will pop up to allow you to load your `.srt` subtitles file.

### 3. Choose where you would like to save your generated `.wav` audio file.
Set the output directory for the dubbing/voiceover.

### 4. Conversion will start - be patient, as the process can take up to a few hours for longer video content :)
