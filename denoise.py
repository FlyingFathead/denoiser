import sys
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

def process_audio(input_file):
    # Load the pretrained DNS64 model
    model = pretrained.dns64().cuda()

    # Load your audio file
    wav, sr = torchaudio.load(input_file)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)

    # Denoise the audio
    with torch.no_grad():
        denoised_wav = model(wav[None])[0]

    # Save your processed file
    output_file = input_file.replace('.mp3', '_denoised.wav')
    torchaudio.save(output_file, denoised_wav.cpu(), model.sample_rate)
    print(f"Processed audio saved as {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file.mp3>")
        sys.exit(1)
    
    process_audio(sys.argv[1])
