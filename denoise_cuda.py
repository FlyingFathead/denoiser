# denoise_cuda.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/denoiser
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import os
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

# Define chunk size and overlap in seconds
CHUNK_DURATION = 10  # Chunk duration in seconds
OVERLAP_DURATION = 1  # Overlap duration in seconds

def process_audio(input_file):
    print("Loading the pretrained DNS64 model...")
    model = pretrained.dns64().cuda()

    print(f"Loading audio file {input_file}...")
    wav, sr = torchaudio.load(input_file)
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)

    print("Preparing for chunk-based processing...")
    chunk_size = int(model.sample_rate * CHUNK_DURATION)
    overlap_size = int(model.sample_rate * OVERLAP_DURATION)
    total_length = wav.size(1)
    num_chunks = (total_length - overlap_size) // (chunk_size - overlap_size) + 1

    print("Starting denoising process...")
    denoised_wav = torch.Tensor().cuda()
    
    try:
        with torch.no_grad():
            for i in range(num_chunks):
                print(f"Processing chunk {i + 1} / {num_chunks}...")
                start = i * (chunk_size - overlap_size)
                end = start + chunk_size if i < num_chunks - 1 else total_length
                chunk = wav[:, start:end]
                denoised_chunk = model(chunk[None])[0]

                # Combine chunks, handling overlaps
                if i > 0:
                    denoised_wav = torch.cat((denoised_wav[:, :-overlap_size], denoised_chunk), 1)
                else:
                    denoised_wav = torch.cat((denoised_wav, denoised_chunk), 1)

    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting...")
        sys.exit(1)

    # Constructing the output filename based on the input file's extension
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_denoised{ext}"
    print(f"Saving processed audio as {output_file}...")
    torchaudio.save(output_file, denoised_wav.cpu(), model.sample_rate)
    print("Processing complete.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python denoise_cuda.py <input_file>")
        sys.exit(1)
    
    process_audio(sys.argv[1])