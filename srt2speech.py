import os
import numpy as np
import torch
import asyncio
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import srt
from scipy.io import wavfile
from tqdm import tqdm
import sys
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv
load_dotenv()

XTTS2_PATH = os.getenv('XTTS2_PATH')
XTTS2_CONFIG = os.getenv('XTTS2_CONFIG')
XTTS2_SAMPLE = os.getenv('XTTS2_SAMPLE')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load XTTS configuration
xtts_config = XttsConfig()
xtts_config.load_json(XTTS2_CONFIG)

# Initialize XTTS model
xtts_model = Xtts.init_from_config(xtts_config)
xtts_model.load_checkpoint(xtts_config, checkpoint_dir=XTTS2_PATH, eval=True)
xtts_model.to(device)

async def stream_xtts2(text, sample_path, speed):
    try:
        outputs = xtts_model.synthesize(
            text,
            xtts_config,
            speaker_wav=sample_path,
            gpt_cond_len=10,
            temperature=0.9,
            top_p=0.7,
            length_penalty=1.0,
            repetition_penalty=6.0,
            language='en',
            speed=speed
        )
        
        audio = outputs['wav']
        sample_rate = xtts_config.audio.sample_rate
        
        return audio, sample_rate
    except Exception as e:
        print(f"Error in stream_xtts2: {e}")
        return None, None

def parse_srt(srt_file):
    with open(srt_file, 'r', encoding='utf-8') as f:
        return list(srt.parse(f))

async def generate_audio_for_subtitle(subtitle, sample_path, target_duration):
    speed = 1.2
    audio_data, sample_rate = await stream_xtts2(subtitle.content, sample_path, speed)
    
    if audio_data is not None:
        current_duration = len(audio_data) / sample_rate * 1000  # duration in milliseconds
        
        while current_duration > target_duration and speed < 1.8:
            speed += 0.1
            audio_data, sample_rate = await stream_xtts2(subtitle.content, sample_path, speed)
            current_duration = len(audio_data) / sample_rate * 1000
        
        # If the audio is still too long, truncate it
        if current_duration > target_duration:
            audio_data = audio_data[:int(target_duration / 1000 * sample_rate)]
        
        return audio_data, sample_rate, speed
    return None, None, speed

async def process_srt_file(srt_file, output_file, sample_path):
    subtitles = parse_srt(srt_file)
    full_audio = np.array([], dtype=np.float32)
    sample_rate = None

    # Create progress bar
    pbar = tqdm(total=len(subtitles), desc="Processing subtitles", file=sys.stdout)

    for subtitle in subtitles:
        start_time = subtitle.start.total_seconds() * 1000
        end_time = subtitle.end.total_seconds() * 1000
        target_duration = end_time - start_time
        
        # Generate audio for the subtitle
        subtitle_audio, current_sample_rate, used_speed = await generate_audio_for_subtitle(subtitle, sample_path, target_duration)
        
        if subtitle_audio is not None:
            if sample_rate is None:
                sample_rate = current_sample_rate
            
            # Add silence before the subtitle if needed
            if start_time > len(full_audio) / sample_rate * 1000:
                silence_duration = int((start_time - len(full_audio) / sample_rate * 1000) * sample_rate / 1000)
                full_audio = np.concatenate([full_audio, np.zeros(silence_duration, dtype=np.float32)])
            
            # Add the subtitle audio
            full_audio = np.concatenate([full_audio, subtitle_audio])
            
            # Update progress bar
            pbar.set_postfix({"Speed": f"{used_speed:.2f}"}, refresh=True)
            pbar.update(1)
            pbar.refresh()

    # Close progress bar
    pbar.close()
    
    # Export the final audio
    try:
        wavfile.write(output_file, sample_rate, (full_audio * 32767).astype(np.int16))
        print(f"\nSuccessfully saved output to {output_file}")
    except Exception as e:
        print(f"\nError saving output file: {e}")

def select_srt_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select SRT file",
        filetypes=[("SRT files", "*.srt"), ("All files", "*.*")]
    )
    return file_path

def select_output_directory():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(
        title="Select output directory"
    )
    return directory

async def main():
    print("Please select the input SRT file.")
    srt_file = select_srt_file()
    if not srt_file:
        print("No SRT file selected. Exiting.")
        return

    print("Please select the output directory.")
    output_directory = select_output_directory()
    if not output_directory:
        print("No output directory selected. Exiting.")
        return

    # Generate output file name based on input file
    output_file = os.path.join(output_directory, os.path.splitext(os.path.basename(srt_file))[0] + "_dubbed.wav")
    
    print(f"Input SRT file: {srt_file}")
    print(f"Output WAV file: {output_file}")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    await process_srt_file(srt_file, output_file, XTTS2_SAMPLE)
    
    if os.path.exists(output_file):
        print(f"Dubbing completed.")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print(f"Error: Output file {output_file} was not created.")

if __name__ == "__main__":
    asyncio.run(main())
