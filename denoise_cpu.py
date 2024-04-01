# denoise_cpu.py

import sys
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

def process_audio(input_file):
    print("Loading the pretrained DNS64 model...")
    model = pretrained.dns64()

    print(f"Loading audio file {input_file}...")
    wav, sr = torchaudio.load(input_file)
    
    print("Converting audio for processing...")
    wav = convert_audio(wav, sr, model.sample_rate, model.chin).half()

    print("Converting model to half precision...")
    model = model.half()

    # Assuming stereo audio - adjust if your audio is mono or has a different number of channels
    print("Starting denoising process...")
    with torch.no_grad():
        # Processing the whole file at once; for very large files you might process in chunks
        denoised_wav = model(wav[None])[0]

    output_file = input_file.replace('.mp3', '_denoised.wav')
    print(f"Saving processed audio as {output_file}...")
    torchaudio.save(output_file, denoised_wav.float().cpu(), model.sample_rate)
    print("Processing complete.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python denoise_cpu.py <input_file.mp3>")
        sys.exit(1)

    process_audio(sys.argv[1])
