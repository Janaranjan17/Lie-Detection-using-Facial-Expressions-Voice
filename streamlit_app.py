import streamlit as st
import subprocess
import threading
import os
import librosa
import soundfile as sf
from io import BytesIO
import tempfile
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time

def extract_audio(video_file, output_audio_file="temp_audio.wav"):
    """Extracts audio from the video file using librosa and saves it as a temporary .wav file."""
    try:
        # Load the audio data from the video file using librosa
        audio_data, sr = librosa.load(video_file, sr=None)
        # Save the audio data to a temporary file using soundfile
        sf.write(output_audio_file, audio_data, sr)
        return output_audio_file, sr
    except Exception as e:
        st.error(f"Audio extraction error: {e}")
        return None, None

def run_subprocess(command, progress_bar, progress_text):
    """Runs a command in a subprocess and returns the output."""
    try:
        process = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Read output in real-time and update progress bar
        total_output = ""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                total_output += output
                progress_text.write(output)  # Display live output

                # Try to estimate progress based on the output (example: assuming progress is indicated by percentages)
                if "%" in output:
                    try:
                        percent_complete = float(output.split("%")[0].strip())
                        progress_bar.progress(percent_complete / 100)
                    except ValueError:
                        pass

        stdout, stderr = total_output, process.stderr.read()

        if stderr:
            st.error(f"Subprocess Error:\n{stderr}")
        return stdout

    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def display_video(video_path, video_placeholder):
    """Display video in Streamlit."""
    try:
        video_bytes = open(video_path, 'rb').read()
        video_placeholder.video(video_bytes)
    except FileNotFoundError:
        st.error(f"Video file not found: {video_path}")
    except Exception as e:
        st.error(f"Error displaying video: {e}")

def analyze_audio(audio_file):
    """Analyzes the audio file using check_truth_success.py."""
    try:
        # Run check_truth_success.py and capture output
        command = ["python", "check_truth_success_png.py", audio_file]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        st.text(output)  # Display captured output

        # Attempt to display plot
        if os.path.exists("truth_chart.png"):  # Check if plot exists
            img = Image.open("truth_chart.png")
            return img
        else:
            st.error("Plot 'truth_chart.png' was not generated. Ensure check_truth_success.py generates the plot.")
            return None

    except subprocess.CalledProcessError as e:
        st.error(f"Error during audio analysis: {e.stderr}")
        return None
    except FileNotFoundError as e:
        st.error(f"Error: check_truth_success.py not found or missing dependencies.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during audio analysis: {e}")
        return None

def main():
    st.title("Video and Audio Analysis Tool")

    # File uploader for video input
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Placeholders for video and audio output
        video_placeholder = st.empty()
        audio_placeholder = st.empty()

        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(video_file.read())
            temp_video_path = temp_video_file.name

        # Display the uploaded video
        video_placeholder.video(video_file)

        # Add a button to trigger the analysis
        if st.button("Start Analysis"):
            with st.spinner("Extracting audio..."):
                # Extract audio from the uploaded video file
                temp_audio_file, _ = extract_audio(temp_video_path)

            if not temp_audio_file:
                st.error("Audio extraction failed. Aborting.")
                return

            # Create placeholders for progress bars and live output
            intercept_progress_bar = st.progress(0)
            intercept_progress_text = st.empty()
            check_truth_progress_bar = st.progress(0)
            check_truth_progress_text = st.empty()

            # Run intercept.py to generate the processed video
            intercept_command = [
                "python",
                "intercept.py",
                "-i",
                temp_video_path,
                "--second",
                "0",
            ]

            # Create threads for each subprocess
            intercept_thread = threading.Thread(
                target=run_subprocess,
                args=(intercept_command, intercept_progress_bar, intercept_progress_text)
            )

            # Run audio analysis and display results
            with st.spinner("Analyzing audio..."):
                audio_plot = analyze_audio(temp_audio_file)
                if audio_plot:
                    audio_placeholder.image(audio_plot, caption="Audio Analysis Results", use_column_width=True)

            # Start video analysis
            intercept_thread.start()

            # Wait for the threads to complete
            intercept_thread.join()

            # Display the processed video
            output_video_path = "output_video.avi"
            display_video(output_video_path, video_placeholder)  # Update video placeholder

            # Clean up temporary files
            os.remove(temp_audio_file)
            os.remove(temp_video_path)

if __name__ == "__main__":
    main()
