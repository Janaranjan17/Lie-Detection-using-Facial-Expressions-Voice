import subprocess
import threading
import shlex  # For properly splitting commands with arguments
import os
import librosa
import soundfile as sf
import argparse

def extract_audio(video_file, output_audio_file="temp_audio.wav"):
    """Extracts audio from the video file using librosa and saves it as a temporary .wav file."""
    try:
        # Load the audio data from the video file using librosa
        audio_data, sr = librosa.load(video_file, sr=None)
        # Save the audio data to a temporary file using soundfile
        sf.write(output_audio_file, audio_data, sr)
        print(f"Audio extracted from {video_file} and saved as {output_audio_file}")
        return output_audio_file, sr
    except Exception as e:
        print(f"Audio extraction error: {e}")
        return None, None

def run_subprocess(command):
    """Runs a command in a subprocess and prints the output."""
    try:
        process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if stdout:
            print(f"Stdout:\n{stdout}")
        if stderr:
            print(f"Stderr:\n{stderr}")
        if process.returncode != 0:
            print(f"Command failed with return code: {process.returncode}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run intercept.py and check_truth_success.py concurrently using subprocess.")
    parser.add_argument("video_file", type=str, help="Path to the video file")
    args = parser.parse_args()

    video_file = args.video_file

    # Extract audio from the video file
    temp_audio_file, _ = extract_audio(video_file)

    if not temp_audio_file:
        print("Audio extraction failed. Aborting.")
        return

    # Define the commands to run
    intercept_command = [
        "python",
        "intercept.py",
        "-i",
        video_file,
        "--second",
        "0",
    ]

    check_truth_command = [
        "python",
        "check_truth_success.py",
        temp_audio_file,
    ]


    # Create threads for each subprocess
    intercept_thread = threading.Thread(target=run_subprocess, args=(intercept_command,))
    check_truth_thread = threading.Thread(target=run_subprocess, args=(check_truth_command,))

    # Start the threads
    intercept_thread.start()
    check_truth_thread.start()

    # Wait for the threads to complete
    intercept_thread.join()
    check_truth_thread.join()

    # Clean up the temporary audio file
    os.remove(temp_audio_file)
    print("Temporary audio file cleaned up.")

if __name__ == "__main__":
    main()
