# Install required libraries
!pip install SpeechRecognition pydub gTTS

import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from google.colab import files
import nltk
import os
import IPython.display as ipd

# Download necessary NLTK data for NLP tasks
nltk.download('punkt')
nltk.download('stopwords')

# Function for Text-to-Speech using gTTS
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("/content/temp_audio.mp3")
    ipd.Audio("/content/temp_audio.mp3")  # Play the speech in the notebook
    return "/content/temp_audio.mp3"

# Function to listen to audio file
def listen_to_audio_from_file(file_path):
    recognizer = sr.Recognizer()
    
    # Convert audio to WAV format if it's not already in that format
    if not file_path.endswith('.wav'):
        sound = AudioSegment.from_file(file_path)
        sound = sound.set_channels(1).set_frame_rate(16000)
        wav_file = "/content/temp_audio.wav"
        sound.export(wav_file, format="wav")
        file_path = wav_file
    
    # Recognize the speech from the audio file
    with sr.AudioFile(file_path) as source:
        print("Processing audio file...")
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            print("Recognized text:", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the audio.")
            return ""
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return ""

# Process the text from the recognized speech
def process_text(text):
    if text:
        # Tokenization and removing stopwords
        words = nltk.word_tokenize(text)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        print("Processed Text:", ' '.join(filtered_words))

        # Example of a basic NLP operation: check if the word "hello" is in the speech
        if "hello" in filtered_words:
            speak("Hello, how can I assist you?")
        else:
            speak("I heard you say: " + ' '.join(filtered_words))

# Main function to upload audio and process it
def main():
    uploaded = files.upload()  # Upload audio file
    for file_name in uploaded.keys():
        print(f"Uploaded file: {file_name}")
        text = listen_to_audio_from_file(file_name)
        process_text(text)

# Run the main function
main()
