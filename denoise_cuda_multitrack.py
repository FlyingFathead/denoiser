# denoise_cuda_multitrack.py

version_number = "0.03"

# (requires `ffprobe` and `ffmpeg`)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/FlyingFathead/denoiser
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import subprocess
import sys
import os
import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

# Define chunk size and overlap in seconds for denoising
CHUNK_DURATION = 10  # Chunk duration in seconds
OVERLAP_DURATION = 1  # Overlap duration in seconds

# Define the directory for processed audio tracks
PROCESSED_AUDIO_DIR = "processed_audio"

# directory checker
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_command(command, get_output=False):
    result = subprocess.run(command, capture_output=True, shell=True, text=True, check=True)
    if get_output:
        return result.stdout.strip()
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

def get_audio_metadata(input_file):
    command = f"ffprobe -v error -select_streams a:0 -show_entries stream=channels,channel_layout -of default=noprint_wrappers=1 {input_file}"
    output = run_command(command, get_output=True)
    lines = output.split('\n')
    metadata = {line.split('=')[0]: line.split('=')[1] for line in lines}
    return int(metadata['channels']), metadata.get('channel_layout', '')

def extract_channels(input_file, num_channels):
    ensure_directory_exists(PROCESSED_AUDIO_DIR)
    channel_files = []
    for i in range(num_channels):
        output_file = os.path.join(PROCESSED_AUDIO_DIR, f"{os.path.splitext(os.path.basename(input_file))[0]}_channel{i}.wav")
        print(f"Extracting channel {i} into {output_file}")
        run_command(f"ffmpeg -i {input_file} -filter_complex 'pan=mono|c0=c{i}' {output_file}")
        channel_files.append(output_file)
    return channel_files

def process_audio(input_file, output_file):
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

    print(f"Saving processed audio as {output_file}...")
    torchaudio.save(output_file, denoised_wav.cpu(), model.sample_rate)

def denoise_channels(channel_files):
    denoised_files = []
    for file in channel_files:
        denoised_file = f"{os.path.splitext(file)[0]}_denoised.wav"
        process_audio(file, denoised_file)
        denoised_files.append(denoised_file)
    return denoised_files

def recombine_channels(denoised_files, channel_layout, input_file):
    print("Recombining channels into stereo mix...")
    mixdown_file = f"{os.path.splitext(input_file)[0]}_denoised_stereo.wav"

    # Construct the filter_complex string dynamically based on channel layout and file list
    inputs_str = ' '.join(f"-i {file}" for file in denoised_files)
    mix_str = ''.join(f"[{i}:a]" for i in range(len(denoised_files)))
    mixdown_cmd = f"ffmpeg {inputs_str} -filter_complex '{mix_str} amix=inputs={len(denoised_files)}:duration=longest' {mixdown_file}"
    
    run_command(mixdown_cmd)

# preprocess the audio
def preprocess_audio(input_file):
    check_cuda_availability()
    
    try:
        print(f"Starting preprocessing for {input_file}...")
        num_channels, channel_layout = get_audio_metadata(input_file)
        print(f"Found {num_channels} channels with layout '{channel_layout}'.")

        if not num_channels:
            print("No audio channels found. Exiting.")
            sys.exit(1)

        channel_files = extract_channels(input_file, num_channels)
        denoised_files = denoise_channels(channel_files)

        # Only attempt recombination if we successfully processed some files.
        if denoised_files:
            recombine_channels(denoised_files, channel_layout, input_file)
            print("Processing complete.")
        else:
            print("No channels were processed. Exiting.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        sys.exit(1)

# check to see that CUDA is available
def check_cuda_availability():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python denoise_cuda_multitrack.py <input_file>")
        sys.exit(1)
    preprocess_audio(sys.argv[1])


# ~~~
# old 
# ~~~

# def run_command(command, get_output=False):
#     result = subprocess.run(command, capture_output=True, shell=True, text=True, check=True)
#     if get_output:
#         return result.stdout.strip()
#     if result.returncode != 0:
#         print(f"Error: {result.stderr}")
#         sys.exit(1)

# def analyze_audio_channels(input_file):
#     command = f"ffprobe -v error -select_streams a:0 -show_entries stream=channels -of default=noprint_wrappers=1:nokey=1 {input_file}"
#     num_channels = int(run_command(command, get_output=True))
#     return num_channels

# def extract_channels(input_file, num_channels):
#     channel_files = []
#     for i in range(num_channels):
#         output_file = f"{os.path.splitext(input_file)[0]}_channel{i}.wav"
#         print(f"Extracting channel {i} into {output_file}")
#         run_command(f"ffmpeg -i {input_file} -filter_complex 'pan=mono|c0=c{i}' {output_file}")
#         channel_files.append(output_file)
#     return channel_files

# def process_audio(input_file, output_file):
#     print("Loading the pretrained DNS64 model...")
#     model = pretrained.dns64().cuda()

#     print(f"Loading audio file {input_file}...")
#     wav, sr = torchaudio.load(input_file)
#     wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)

#     print("Preparing for chunk-based processing...")
#     chunk_size = int(model.sample_rate * CHUNK_DURATION)
#     overlap_size = int(model.sample_rate * OVERLAP_DURATION)
#     total_length = wav.size(1)
#     num_chunks = (total_length - overlap_size) // (chunk_size - overlap_size) + 1

#     print("Starting denoising process...")
#     denoised_wav = torch.Tensor().cuda()
    
#     with torch.no_grad():
#         for i in range(num_chunks):
#             print(f"Processing chunk {i + 1} / {num_chunks}...")
#             start = i * (chunk_size - overlap_size)
#             end = start + chunk_size if i < num_chunks - 1 else total_length
#             chunk = wav[:, start:end]
#             denoised_chunk = model(chunk[None])[0]

#             # Combine chunks, handling overlaps
#             if i > 0:
#                 denoised_wav = torch.cat((denoised_wav[:, :-overlap_size], denoised_chunk), 1)
#             else:
#                 denoised_wav = torch.cat((denoised_wav, denoised_chunk), 1)

#     print(f"Saving processed audio as {output_file}...")
#     torchaudio.save(output_file, denoised_wav.cpu(), model.sample_rate)

# def denoise_channels(channel_files):
#     denoised_files = []
#     for file in channel_files:
#         denoised_file = f"{os.path.splitext(file)[0]}_denoised.wav"
#         process_audio(file, denoised_file)
#         denoised_files.append(denoised_file)
#     return denoised_files

# # def recombine_channels(denoised_files, input_file):
# #     print("Recombining channels into stereo mix...")
# #     mixdown_file = f"{os.path.splitext(input_file)[0]}_denoised_stereo.wav"
# #     files_string = ' '.join([f"-i {file}" for file in denoised_files])
# #     filter_complex = ' '.join([f"[{i}]pan=stereo|c0=c0|c1=c0[a{i}];" for i in range(len(denoised_files))])
# #     filter_complex = f"{filter_complex}[a0][a1]amerge=inputs={len(denoised_files)}"
# #     run_command(f"ffmpeg {files_string} -filter_complex '{filter_complex}' {mixdown_file}")

# def recombine_channels(denoised_files, input_file):
#     print("Recombining channels into stereo mix...")
#     fl, fr, fc, lfe, sl, sr = denoised_files  # Assuming the order matches FL, FR, FC, LFE, SL, SR
#     mixdown_file = f"{os.path.splitext(input_file)[0]}_denoised_stereo.wav"
#     run_command(f"""
#         ffmpeg -i {fl} -i {fr} -i {fc} -i {sl} -i {sr} -filter_complex \
#         "[0:a][1:a]amerge=inputs=2[FLFR]; \
#          [2:a]asplit[FC1][FC2]; \
#          [3:a]asplit[SL1][SL2]; \
#          [4:a]asplit[SR1][SR2]; \
#          [FLFR][FC1][FC2][SL1][SL2][SR1][SR2]amix=inputs=7:weights=1 1 0.7 0.7 0.5 0.5 0.5 0.5[a]" \
#         -map "[a]" {mixdown_file}
#     """)

# def preprocess_audio(input_file):
#     print(f"Starting preprocessing for {input_file}...")
#     num_channels = analyze_audio_channels(input_file)
#     channel_files = extract_channels(input_file, num_channels)
#     denoised_files = denoise_channels(channel_files)
#     recombine_channels(denoised_files, input_file)
#     print("Processing complete.")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python denoise_cuda_multitrack.py <input_file>")
#         sys.exit(1)
#     preprocess_audio(sys.argv[1])