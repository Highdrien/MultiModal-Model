from faster_whisper import WhisperModel
import os
import torch
import soundfile as sf
import time
import requests

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def transcribe_audio(audio_path, output_dir, segment_duration,beam_size=3):
    """
    Transcribes the audio file located at `audio_path` into text segments of duration `segment_duration`.
    The transcribed segments are saved as separate WAV files in the `output_dir` directory.
    The final transcription is saved as a text file named 'transcription.txt' in the `output_dir` directory.

    Args:
        audio_path (str): The path to the audio file.
        output_dir (str): The directory where the transcribed segments and the final transcription will be saved.
        segment_duration (float): The duration of each segment in milliseconds.

    Returns:
        str: The total transcription as a single string.
    """
    model_size = "medium"
    model = WhisperModel(model_size, device="cuda", compute_type="float16") #load the whisper model
    print("Model loaded")
    if not os.path.exists(output_dir): # Create the output directory if it doesn't exist
        os.makedirs(output_dir)

    audio, sample_rate = sf.read(audio_path) # Load the audio file
    total_duration = len(audio) / sample_rate * 1000  # Convert to milliseconds
    total_transcription = "" # The total transcription as a single string

    start = 0
    segment_number = 1

    while start < total_duration: # Transcribe each segment
        end = min(start + segment_duration, total_duration) # The end of the segment in milliseconds
        start_frame = int(start / 1000 * sample_rate) # The start of the segment in frames
        end_frame = int(end / 1000 * sample_rate) # The end of the segment in frames
        segment = audio[start_frame:end_frame] # The segment as a numpy array

        segment_path = os.path.join(output_dir, f"segment_{segment_number}.wav") # The path to the segment's WAV file
        sf.write(segment_path, segment, sample_rate) # Save the segment as a WAV file

        print("Transcribing segment", segment_number, "out of", int(total_duration / segment_duration)) # Print progress

        segments, info = model.transcribe(segment_path, beam_size=beam_size) # Transcribe the segment
        #empty cache
        torch.cuda.empty_cache()
        for segment in segments:
            print(segment.text)
            total_transcription += segment.text # Add the segment's transcription to the total transcription

        start = end
        segment_number += 1

    # Save the transcription to a text file
    transcription_file_path = os.path.join(output_dir, 'transcription.txt')
    with open(transcription_file_path, 'w', encoding='utf-8') as f:
        f.write(total_transcription)
        print("Transcription written in file!")

    return total_transcription

if __name__ == "__main__":
    
    audio_path = "C:/Users/Yanis/Downloads/test_2.wav" # The path to the audio file
    output_directory = "C:/Users/Yanis/Downloads/transcription_output" # The directory where the transcribed segments and the final transcription will be saved (will be created if it doesn't exist)
    segment_duration = 30000  # The duration of each segment in milliseconds
    beam_size = 3 # The beam size used by the model (if too high, the model will run out of memory)

    # Measure time of transcribing audio
    start_time = time.time()
    transcribe_audio(audio_path, output_directory, segment_duration,beam_size) # Transcribe the audio file
    end_time = time.time()
    print("Time of transcription:", end_time - start_time)
