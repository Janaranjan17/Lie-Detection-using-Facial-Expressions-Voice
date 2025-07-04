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

# Adjust path to include the directory containing the analysis scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get directory of current script
sys.path.append(os.path.join(SCRIPT_DIR, 'audio_video_truth', 'Truthsayer-master'))

# Import the audio and video analysis functions from their respective files
try:
    from check_truth_success import main as audio_main, analyze_audio_file, plot_results as audio_plot_results, analyze_truth_level
except ImportError as e:
    print(f"Error importing audio analysis functions: {e}")
    audio_main = None
    analyze_audio_file = None
    audio_plot_results = None
    analyze_truth_level = None

try:
    from intercept import main as video_main, process as video_process, add_text, draw_on_frame, find_face_and_hands,get_bpm_tells,is_blinking,get_blink_tell,check_hand_on_face,get_avg_gaze,detect_gaze_change,get_lip_ratio,get_mood,add_truth_meter,get_face_relative_area,get_area, new_tell, decrement_tells, chart_setup, TEXT_HEIGHT, EPOCH, MAX_FRAMES
except ImportError as e:
    print(f"Error importing video analysis functions: {e}")
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

# -----------------------------------------------------------------------------
# Shared Configuration and Helper Functions
# -----------------------------------------------------------------------------

AUDIO_SEGMENT_LENGTH = 5
RECORDING_FILENAME = None
tells = {}  # Initialize tells dictionary

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

def cleanup_temp_files(temp_audio_file):
    """Cleans up temporary files after analysis."""
    try:
        os.remove(temp_audio_file)
        print(f"Successfully removed temporary audio file: {temp_audio_file}")
    except Exception as e:
        print(f"Error removing temporary audio file: {e}")

# -----------------------------------------------------------------------------
# Modified Analysis Functions
# -----------------------------------------------------------------------------

def video_analysis_wrapper(video_file, landmarks, bpm, flip, ttl, record,recording):
    """Wrapper to call video analysis and handle setup."""
    global tells

    if video_main is None or video_process is None:
        print("Video analysis not available.")
        return

    # Add the variable for the filename
    # Add the variable recording for the global scope
    
    # Add your intercept.py logic here, using the arguments passed to the function
    try:
        # Initialize video capture and other setup as done in the original main()
        # Set value for the RECORDING_FILENAME
        RECORDING_FILENAME = str(datetime.now()).replace('.','').replace(':','') + '.avi'
        DRAW_LANDMARKS = landmarks
        BPM_CHART = bpm
        FLIP = flip
        TELL_MAX_TTL = ttl
        RECORD = record
        #print (DRAW_LANDMARKS,BPM_CHART,FLIP,TELL_MAX_TTL,RECORD)

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
                    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
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
                    cv2.imshow('face', image)
                    if RECORD:
                        recording.write(image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                #if SECOND:
                #    cap2.release()
                if RECORD:
                    recording.release()
                cv2.destroyAllWindows()
                print("Time taken for this process ",time.time()-start)

    except Exception as e:
        print(f"Error in video analysis wrapper: {e}")

def audio_analysis_wrapper(audio_file, sr):
    """Wrapper to call audio analysis and handle setup."""
    if audio_main is None or analyze_audio_file is None:
        print("Audio analysis not available.")
        return

    try:
        # Call the imported audio analysis function
        emotion_labels, emotion_scores, truth_levels = analyze_audio_file(audio_file)
        # Call the imported audio plotting function
        audio_plot_results(emotion_labels, emotion_scores, truth_levels)

    except Exception as e:
        print(f"Error in audio analysis wrapper: {e}")

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze video for facial expressions and audio for truthfulness.")
    parser.add_argument('video_file', type=str, help='Path to the video file')
    parser.add_argument('--landmarks', '-l', action='store_true', help='Enable face landmarks')
    parser.add_argument('--bpm', '-b', action='store_true', help='Enable BPM chart')
    parser.add_argument('--flip', '-f', action='store_true', help='Flip video')
    parser.add_argument('--ttl', '-t', type=int, default=30, help='TTL for tells')
    parser.add_argument('--record', '-r', action='store_true', help='Record video')
    args = parser.parse_args()
    # add the variable recording over the video to be added
    recording = None

    # 1. Extract audio
    temp_audio_file, sr = extract_audio(args.video_file)

    if temp_audio_file:  # If successful audio extraction
        # 2. Run the video and audio analysis
        video_thread = threading.Thread(target=video_analysis_wrapper, args=(args.video_file, args.landmarks, args.bpm, args.flip, args.ttl, args.record,recording))
        audio_thread = threading.Thread(target=audio_analysis_wrapper, args=(temp_audio_file, sr))

        video_thread.start()
        audio_thread.start()

        video_thread.join()
        audio_thread.join()

        # 3. Clean up
        cleanup_temp_files(temp_audio_file)
    else:
        print("Aborting analysis due to audio extraction failure.")

if __name__ == "__main__":
    main()
