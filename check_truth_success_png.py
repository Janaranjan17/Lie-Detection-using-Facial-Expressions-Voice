import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from transformers import pipeline

# Load the emotion recognition model
emotion_model = pipeline("audio-classification", model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")

# Hypothetical function for truth level analysis
def analyze_truth_level(audio_data, sr):
    # This is a placeholder for actual lie detection logic.
    # For simplicity, let's simulate truth levels based on audio characteristics.
    # In practice, you would implement or call a lie detection model here.
    truth_levels = np.random.rand(len(audio_data) // sr)  # Random values for demonstration
    return truth_levels

def analyze_audio_file(audio_file):
    # Load the audio file using librosa
    audio_data, sr = librosa.load(audio_file, sr=None)

    # Save the audio data to a temporary file using soundfile
    temp_audio_file = "temp_audio.wav"
    sf.write(temp_audio_file, audio_data, sr)
    print("Audio file loaded and saved for transcription.")

    # Analyze emotions every 5 seconds
    emotion_labels = []
    emotion_scores = []
    truth_levels = []

    duration = len(audio_data) / sr
    num_segments = int(np.ceil(duration / 5))  # Number of 5-second segments

    for i in range(num_segments):
        start_sample = i * 5 * sr
        end_sample = min((i + 1) * 5 * sr, len(audio_data))
        segment = audio_data[start_sample:end_sample]

        # Analyze emotions in the current segment
        emotions = emotion_model(temp_audio_file)
        emotion_labels.append([emotion['label'] for emotion in emotions])
        emotion_scores.append([emotion['score'] for emotion in emotions])

        # Analyze truth level for the current segment
        truth_level_segment = analyze_truth_level(segment, sr)
        truth_levels.append(np.mean(truth_level_segment))  # Average truth level for this segment

    return emotion_labels, emotion_scores, truth_levels

def plot_results(emotion_labels, emotion_scores, truth_levels, output_filename="truth_chart.png"):
    """Plot the detected emotions and truth levels and save the figure."""
    
    # Plotting emotions
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    for i in range(len(emotion_labels)):
        plt.bar(emotion_labels[i], emotion_scores[i], label=f'Segment {i+1}')
    
    plt.title("Detected Emotions in Audio")
    plt.xlabel("Emotions")
    plt.ylabel("Scores")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Plotting truth levels
    plt.subplot(2, 1, 2)
    plt.plot(range(len(truth_levels)), truth_levels, marker='o', color='red')
    
    plt.title("Truth Level Over Time")
    plt.xlabel("5-Second Segments")
    plt.ylabel("Truth Level")
    plt.ylim(0, 1)
    
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_filename)
    plt.close()  # Close the plot to free memory
    print(f"Plot saved to {output_filename}")

def main():
    # Argument parser for audio file input
    parser = argparse.ArgumentParser(description="Analyze audio file for emotions and truthfulness.")
    parser.add_argument('audio_file', type=str, help='Path to the audio file to analyze')
    parser.add_argument('--output', type=str, default="truth_chart.png", help='Output filename for the plot (default: truth_chart.png)')
    args = parser.parse_args()

    # Analyze the provided audio file
    emotion_labels, emotion_scores, truth_levels = analyze_audio_file(args.audio_file)

    # Print the results
    for i in range(len(emotion_labels)):
        for label, score in zip(emotion_labels[i], emotion_scores[i]):
            print(f"Segment {i+1} - Emotion: {label}, Score: {score:.2f}")

    # Plot and save the results
    plot_results(emotion_labels, emotion_scores, truth_levels, args.output)

if __name__ == '__main__':
    main()
