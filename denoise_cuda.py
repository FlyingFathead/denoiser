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
MIX_LEVEL = 0.5  # 1.0 = all denoised, 0 = all original, 0.85 is a strong denoise, tweak as desired

def process_audio(input_file, mix_level=MIX_LEVEL):
    print(f">>> Wet/dry mix level: {mix_level:.2f}  (1 = all denoised, 0 = original)")    
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

                # Apply wet/dry mix
                # Ensure both are same length (might not be at the end!)
                min_len = min(chunk.shape[1], denoised_chunk.shape[1])
                chunk = chunk[:, :min_len]
                denoised_chunk = denoised_chunk[:, :min_len]

                mixed_chunk = mix_level * denoised_chunk + (1 - mix_level) * chunk

                # Combine chunks, handling overlaps
                if i > 0:
                    denoised_wav = torch.cat((denoised_wav[:, :-overlap_size], mixed_chunk), 1)
                else:
                    denoised_wav = torch.cat((denoised_wav, mixed_chunk), 1)


    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting...")
        sys.exit(1)

    # Constructing the output filename based on the input file's extension
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_denoised{ext}"
    print(f"Saving processed audio as {output_file}...")
    torchaudio.save(output_file, denoised_wav.cpu(), model.sample_rate)
    print("Processing complete.")

if __name__ == "__main__":
    # Accept either 1 or 2 arguments after the script name
    #   argv[1] = input file  (mandatory)
    #   argv[2] = mix level   (optional, 0-1 float)
    if len(sys.argv) not in (2, 3):
        print("Usage: python denoise_cuda.py <input_file> [mix_level 0.0-1.0]")
        sys.exit(1)

    input_path = sys.argv[1]

    # Use provided mix level or fall back to constant
    mix = float(sys.argv[2]) if len(sys.argv) == 3 else MIX_LEVEL

    # Clamp mix just in case someone passes garbage
    mix = max(0.0, min(mix, 1.0))

    process_audio(input_path, mix)
    