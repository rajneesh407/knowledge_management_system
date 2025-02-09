import base64
from base64 import b64decode
from PIL import Image
import io
import yt_dlp
import ffmpeg
import speech_recognition as sr
import cv2
import os


def display_base64_image(base64_code):
    """
    Function to display the image from base64 version.
    """
    image_data = base64.b64decode(base64_code)
    return Image.open(io.BytesIO(image_data))


def parse_docs_for_images_and_texts(docs):
    b64 = []
    text = []
    for doc in docs:
        if doc.page_content.strip().isascii():
            text.append(doc)
        else:
            try:
                b64decode(doc.page_content)
                b64.append(doc.page_content)
            except Exception as e:
                print(e)
                text.append(doc)
    return {"images": b64, "texts": text}


def download_youtube_video(youtube_url, collection_path):
    ydl_opts = {"format": "best", "outtmpl": f"{collection_path}\\video.%(ext)s"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        return ydl.prepare_filename(info)


def clean_transcript(raw_text):
    lines = raw_text.split("\n")
    cleaned_lines = []
    for line in lines:
        if "-->" not in line and not line.strip().isdigit():
            cleaned_lines.append(line.strip())
    return " ".join(cleaned_lines)


def download_transcript(video_url):
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "writesubtitles": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": "%(id)s",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        video_id = info["id"]
        subtitle_file = f"{video_id}.en.vtt"
        if os.path.exists(subtitle_file):
            with open(subtitle_file, "r", encoding="utf-8") as f:
                transcript = f.read()
            return clean_transcript(transcript)
        else:
            return None


def transcribe_video(video_path):
    recognizer = sr.Recognizer()
    audio_file = "temp_audio.wav"
    (
        ffmpeg.input(video_path)
        .output(audio_file, format="wav", acodec="pcm_s16le", ac=1, ar="16000")
        .run(overwrite_output=True, quiet=True)
    )

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    return text


def get_video_transcript(video_url, collection_path):
    transcript = download_transcript(video_url)
    if not transcript:
        print("No subtitles found. Transcribing audio...")
        transcript = transcribe_video(
            download_youtube_video(video_url, collection_path)
        )
    return transcript


def transcribe_audio_file(audio_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)

    return text
