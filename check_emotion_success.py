import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from transformers import pipeline

# Load the emotion recognition model
emotion_model = pipeline("audio-classification", model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")

def analyze_audio_file(audio_file):
    # Load the audio file using librosa
    audio_data, sr = librosa.load(audio_file, sr=None)
    
    # Save the audio data to a temporary file using soundfile
    temp_audio_file = "temp_audio.wav"
    sf.write(temp_audio_file, audio_data, sr)
    print("Audio file loaded and saved for transcription.")

    # Analyze emotions
    emotions = emotion_model(temp_audio_file)
    
    # Extract emotion labels and scores
    emotion_labels = [emotion['label'] for emotion in emotions]
    emotion_scores = [emotion['score'] for emotion in emotions]
    
    return emotion_labels, emotion_scores

def plot_emotions(emotion_labels, emotion_scores):
    """Plot the emotions detected in the audio."""
    plt.figure(figsize=(10, 5))
    plt.bar(emotion_labels, emotion_scores, color='skyblue')
    plt.title("Detected Emotions in Audio")
    plt.xlabel("Emotions")
    plt.ylabel("Scores")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def main():
    # Argument parser for audio file input
    parser = argparse.ArgumentParser(description="Analyze audio file for emotions.")
    parser.add_argument('audio_file', type=str, help='Path to the audio file to analyze')
    args = parser.parse_args()

    # Analyze the provided audio file
    emotion_labels, emotion_scores = analyze_audio_file(args.audio_file)

    # Print the results
    for label, score in zip(emotion_labels, emotion_scores):
        print(f"Emotion: {label}, Score: {score:.2f}")

    # Plot the emotions
    plot_emotions(emotion_labels, emotion_scores)

if __name__ == '__main__':
    main()