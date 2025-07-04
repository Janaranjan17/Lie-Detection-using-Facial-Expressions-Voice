import streamlit as st
import argparse
import threading
import time
import os
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import mediapipe as mp
import cv2
from io import BytesIO
from PIL import Image
import tempfile
import shutil
import queue
import subprocess  # Import subprocess

# -----------------------------------------------------------------------------
# Global Variables and Helper Functions (From video_display.py)
# -----------------------------------------------------------------------------

AUDIO_SEGMENT_LENGTH = 5
RECORDING_FILENAME = None
tells = {}

def extract_audio(video_file, output_audio_file="temp_audio.wav"):
    """Extracts audio from the video file using librosa and saves it as a temporary .wav file."""
    try:
        # Load the audio data from the video file using librosa
        audio_data, sr = librosa.load(video_file, sr=None)
        # Save the audio data to a temporary file using soundfile
        sf.write(output_audio_file, audio_data, sr)
        st.success(f"Audio extracted from {video_file} and saved as {output_audio_file}")
        return output_audio_file, sr
    except Exception as e:
        st.error(f"Audio extraction error: {e}")
        return None, None

def cleanup_temp_files(temp_audio_file):
    """Cleans up temporary files after analysis."""
    try:
        os.remove(temp_audio_file)
        st.success(f"Successfully removed temporary audio file: {temp_audio_file}")
    except Exception as e:
        st.error(f"Error removing temporary audio file: {e}")

# -----------------------------------------------------------------------------
# Import Truthsayer Modules (Cached) (From video_display.py)
# -----------------------------------------------------------------------------

@st.cache_resource()
def load_truth_sayer_modules(script_dir):
    """Loads Truthsayer modules and caches them for reuse."""
    sys.path.append(os.path.join(script_dir, 'audio_video_truth', 'Truthsayer-master'))

    try:
        from check_truth_success_png import main as audio_main, analyze_audio_file, plot_results as audio_plot_results, analyze_truth_level
    except ImportError as e:
        st.error(f"Error importing audio analysis functions: {e}")
        audio_main = None
        analyze_audio_file = None
        audio_plot_results = None
        analyze_truth_level = None

    try:
        from intercept import main as video_main, process as video_process, add_text, draw_on_frame, find_face_and_hands,get_bpm_tells,is_blinking,get_blink_tell,check_hand_on_face,get_avg_gaze,detect_gaze_change,get_lip_ratio,get_mood,add_truth_meter,get_face_relative_area,get_area, new_tell, decrement_tells, chart_setup, TEXT_HEIGHT, EPOCH, MAX_FRAMES
    except ImportError as e:
        st.error(f"Error importing video analysis functions: {e}")
        video_main = None
        video_process = None
        add_text = None
        draw_on_frame=None
        find_face_and_hands=None
        get_bpm_tells=None
        is_blinking=None
        get_blink_tell=None
        check_hand_on_face=None
        get_avg_gaze=None
        detect_gaze_change=None
        get_lip_ratio=None
        get_mood=None
        add_truth_meter=None
        get_face_relative_area=None
        get_area=None
        new_tell=None
        decrement_tells=None
        chart_setup=None
        TEXT_HEIGHT=None
        EPOCH=None
        MAX_FRAMES=None

    return (audio_main, analyze_audio_file, audio_plot_results, analyze_truth_level,
            video_main, video_process, add_text, draw_on_frame, find_face_and_hands,
            get_bpm_tells, is_blinking, get_blink_tell, check_hand_on_face, get_avg_gaze,
            detect_gaze_change, get_lip_ratio, get_mood, add_truth_meter, get_face_relative_area,
            get_area, new_tell, decrement_tells, chart_setup, TEXT_HEIGHT, EPOCH, MAX_FRAMES)

# -----------------------------------------------------------------------------
# Analysis Function Wrappers (Modified from video_display.py and audio_analysis.py)
# -----------------------------------------------------------------------------

def video_analysis_wrapper(video_file, landmarks, bpm, flip, ttl, record, video_queue, error_queue,
                             video_main, video_process, add_text, draw_on_frame, find_face_and_hands,
                             get_bpm_tells, is_blinking, get_blink_tell, check_hand_on_face, get_avg_gaze,
                             detect_gaze_change, get_lip_ratio, get_mood, add_truth_meter, get_face_relative_area,
                             get_area, new_tell, decrement_tells, chart_setup, TEXT_HEIGHT, EPOCH, MAX_FRAMES):
    """Wrapper to call video analysis and handle setup."""
    global tells, RECORDING_FILENAME

    if video_main is None or video_process is None:
        error_queue.put("Video analysis not available.")
        return

    try:
        # Initialize video capture and other setup as done in the original main()
        RECORDING_FILENAME = str(datetime.now()).replace('.','').replace(':','') + '.avi'
        DRAW_LANDMARKS = landmarks
        BPM_CHART = bpm
        FLIP = flip
        TELL_MAX_TTL = ttl
        RECORD = record
        recording = None

        if BPM_CHART and chart_setup is not None:
            chart_setup()

        calibrated = False
        calibration_frames = 0

        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            with mp.solutions.hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7) as hands:

                cap = cv2.VideoCapture(video_file)
                fps = None
                if isinstance(video_file, str) and video_file.find('.') > -1: # from file
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print("FPS:", fps)
                else: # from device
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                if RECORD:
                    FPS_OUT = 10
                    FRAME_SIZE = (int(cap.get(3)), int(cap.get(4)))
                    recording = cv2.VideoWriter(
                        RECORDING_FILENAME, cv2.VideoWriter_fourcc(*'MJPG'), FPS_OUT, FRAME_SIZE)
                start = time.time()
                while cap.isOpened():
                    success, image = cap.read()
                    if not success: break
                    calibration_frames += video_process(image, face_mesh, hands, calibrated, DRAW_LANDMARKS, BPM_CHART, FLIP, fps)
                    calibrated = (calibration_frames >= 120)

                    # Convert the image to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb)

                    # Safely send the image to Streamlit
                    try:
                        video_queue.put(image_pil)
                    except Exception as e:
                        error_queue.put(f"Error sending frame to Streamlit: {e}")
                        break  # Exit loop if there's an error sending to Streamlit...
                    if RECORD and recording:
                        recording.write(image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                if RECORD and recording:
                    recording.release()
                cv2.destroyAllWindows()
                print("Time taken for this process ",time.time()-start)

    except Exception as e:
        error_queue.put(f"Error in video analysis wrapper: {e}")

def analyze_audio(audio_file, audio_placeholder): # Modified
    """Analyzes the audio file using check_truth_success.py."""
    try:
        # Run check_truth_success.py and capture output
        command = ["python", "check_truth_success_png.py", audio_file] #check_truth_success_png
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        st.text(output)  # Display captured output

        # Attempt to display plot
        if os.path.exists("truth_chart.png"):  # Check if plot exists
            img = Image.open("truth_chart.png")
            audio_placeholder.image(img, caption="Audio Analysis Results", use_column_width=True) # Modified to use the placeholder
            return
        else:
            st.error("Plot 'truth_chart.png' was not generated. Ensure check_truth_success.py generates the plot.")
            return

    except subprocess.CalledProcessError as e:
        st.error(f"Error during audio analysis: {e.stderr}")
        return
    except FileNotFoundError as e:
        st.error(f"Error: check_truth_success_png.py not found or missing dependencies.")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during audio analysis: {e}")
        return

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def run_analysis(video_file, landmarks, bpm, flip, ttl, record,
                 audio_main, analyze_audio_file, audio_plot_results, analyze_truth_level,
                 video_main, video_process, add_text, draw_on_frame, find_face_and_hands,
                 get_bpm_tells, is_blinking, get_blink_tell, check_hand_on_face, get_avg_gaze,
                 detect_gaze_change, get_lip_ratio, get_mood, add_truth_meter, get_face_relative_area,
                 get_area, new_tell, decrement_tells, chart_setup, TEXT_HEIGHT, EPOCH, MAX_FRAMES):

    st.subheader("Analysis Results")

    # Placeholders
    video_placeholder = st.empty()  # For showing the video frames
    audio_placeholder = st.empty()  # For the audio analysis plot
    error_placeholder = st.empty()  # For error messages

    # Queues for thread communication
    video_queue = queue.Queue()
    error_queue = queue.Queue()

    # Use a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_audio_file = os.path.join(temp_dir, "temp_audio.wav")

    try:
        # 1. Extract audio
        temp_audio_file, sr = extract_audio(video_file, temp_audio_file)

        if temp_audio_file:
            # 2. Run video analysis in a separate thread
            video_thread = threading.Thread(
                target=video_analysis_wrapper,
                args=(video_file, landmarks, bpm, flip, ttl, record, video_queue, error_queue,
                      video_main, video_process, add_text, draw_on_frame, find_face_and_hands,
                      get_bpm_tells, is_blinking, get_blink_tell, check_hand_on_face, get_avg_gaze,
                      detect_gaze_change, get_lip_ratio, get_mood, add_truth_meter, get_face_relative_area,
                      get_area, new_tell, decrement_tells, chart_setup, TEXT_HEIGHT, EPOCH, MAX_FRAMES)
            )
            video_thread.daemon = True  # Allow the main thread to exit even if this thread is running
            video_thread.start()

            # 3. Analyze audio and display the results
            with st.spinner("Analyzing audio..."):
                analyze_audio(temp_audio_file, audio_placeholder)  # Pass the placeholder to analyze_audio

            # 4. Continuously display video frames
            while video_thread.is_alive() or not video_queue.empty():
                try:
                    image_pil = video_queue.get(timeout=0.1)
                    video_placeholder.image(image_pil, caption='Processed Video', use_column_width=True)
                except queue.Empty:
                    pass  # No frame available yet

        else:
            st.error("Aborting analysis due to audio extraction failure.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    finally:
        # 5. Handle any errors that occurred in the threads
        if not error_queue.empty():
            error_message = error_queue.get()
            st.error(error_message)

        # 6. Clean up
        cleanup_temp_files(temp_audio_file)
        shutil.rmtree(temp_dir)  # Remove the temporary directory

def main():
    st.title("Truth Detection Analysis")

    # Sidebar for settings
    st.sidebar.header("Settings")
    landmarks = st.sidebar.checkbox("Enable Face Landmarks")
    bpm = st.sidebar.checkbox("Enable BPM Chart")
    flip = st.sidebar.checkbox("Flip Video")
    ttl = st.sidebar.slider("TTL for Tells", 10, 60, 30)
    record = st.sidebar.checkbox("Record Video")

    # Video input
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=video_file.name) as temp_video_file:
            temp_video_file.write(video_file.read())
            video_path = temp_video_file.name

        st.video(video_file)

        # Load TruthSayer modules once
        script_dir = os.path.dirname(os.path.abspath(__file__))
        audio_main, analyze_audio_file, audio_plot_results, analyze_truth_level, video_main, video_process, add_text, draw_on_frame, find_face_and_hands, get_bpm_tells, is_blinking, get_blink_tell, check_hand_on_face, get_avg_gaze, detect_gaze_change, get_lip_ratio, get_mood, add_truth_meter, get_face_relative_area, get_area, new_tell, decrement_tells, chart_setup, TEXT_HEIGHT, EPOCH, MAX_FRAMES = load_truth_sayer_modules(script_dir)

        # Button to start analysis
        if st.button("Start Analysis"):
            with st.spinner("Running analysis..."):
                run_analysis(video_path, landmarks, bpm, flip, ttl, record,
                             audio_main, analyze_audio_file, audio_plot_results, analyze_truth_level,
                             video_main, video_process, add_text, draw_on_frame, find_face_and_hands,
                             get_bpm_tells, is_blinking, get_blink_tell, check_hand_on_face, get_avg_gaze,
                             detect_gaze_change, get_lip_ratio, get_mood, add_truth_meter, get_face_relative_area,
                             get_area, new_tell, decrement_tells, chart_setup, TEXT_HEIGHT, EPOCH, MAX_FRAMES)
        # Clean up the temporary video file
        os.remove(video_path)

if __name__ == "__main__":
    main()
