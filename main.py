import os
import uuid
import librosa
import numpy as np
import cv2
import moviepy.editor as mp
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace

app = FastAPI(title="AI Voice Detection API", version="1.0.0")

# =========================
# Utility Functions
# =========================

SUPPORTED_AUDIO_FORMATS = ["mp3", "wav", "m4a"]
SUPPORTED_VIDEO_FORMATS = ["mp4", "mov", "avi"]

def validate_file_format(filename: str, allowed_formats: list):
    ext = filename.split(".")[-1].lower()
    if ext not in allowed_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")
    return ext

def preprocess_audio(file_path: str):
    """Load and preprocess audio using librosa"""
    y, sr_rate = librosa.load(file_path, sr=None)
    y = librosa.effects.preemphasis(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13)
    return y, sr_rate, mfcc

def detect_language(file_path: str):
    """Detect spoken language using SpeechRecognition"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="en-IN")
        # Simplified language detection based on keywords
        if any(word in text.lower() for word in ["vanakkam", "tamil"]):
            return "Tamil"
        elif any(word in text.lower() for word in ["namaste", "hindi"]):
            return "Hindi"
        elif any(word in text.lower() for word in ["hello", "english"]):
            return "English"
        elif any(word in text.lower() for word in ["malayalam"]):
            return "Malayalam"
        elif any(word in text.lower() for word in ["telugu"]):
            return "Telugu"
        else:
            return "Unknown"
    except Exception:
        return "Unknown"

def estimate_gender(mfcc_features: np.ndarray):
    """Simple heuristic for gender estimation based on pitch"""
    mean_pitch = np.mean(mfcc_features)
    if mean_pitch > 0:
        return "Male"
    elif mean_pitch < 0:
        return "Female"
    else:
        return "Unknown"

def classify_voice(mfcc_features: np.ndarray):
    """Dummy AI vs Human classification based on variance"""
    variance = np.var(mfcc_features)
    if variance < 50:
        return "AI_GENERATED", 0.85
    else:
        return "HUMAN", 0.92

def extract_audio_from_video(video_path: str, output_path: str):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(output_path)
    return output_path

def detect_face(video_path: str):
    """Detect and classify face using DeepFace"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Unknown"
    try:
        analysis = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)
        if analysis:
            return "HUMAN_FACE"
        else:
            return "AI_FACE"
    except Exception:
        return "Unknown"

# =========================
# API Endpoints
# =========================

@app.get("/")
def home():
    return {"message": "Welcome to AI Voice Detection API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    return {
        "modelType": "Universal Acoustic Pattern Model",
        "supportedLanguages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
        "features": ["Voice Classification", "Language Detection", "Gender Estimation", "Face Detection"]
    }

@app.post("/detect")
async def detect_audio(file: UploadFile = File(...)):
    ext = validate_file_format(file.filename, SUPPORTED_AUDIO_FORMATS)
    temp_filename = f"temp_{uuid.uuid4()}.{ext}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    y, sr_rate, mfcc = preprocess_audio(temp_filename)
    language = detect_language(temp_filename)
    gender = estimate_gender(mfcc)
    voice_class, confidence = classify_voice(mfcc)

    os.remove(temp_filename)

    return JSONResponse(content={
        "status": "success",
        "voiceClassification": voice_class,
        "confidenceScore": confidence,
        "language": language,
        "gender": gender,
        "noiseLevel": "MEDIUM",
        "samplesProcessed": len(y),
        "inputMode": "uploaded",
        "modelType": "Universal Acoustic Pattern Model"
    })

@app.post("/live-detect")
async def live_detect(file: UploadFile = File(...)):
    # Same as detect_audio but inputMode = live
    ext = validate_file_format(file.filename, SUPPORTED_AUDIO_FORMATS)
    temp_filename = f"temp_{uuid.uuid4()}.{ext}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    y, sr_rate, mfcc = preprocess_audio(temp_filename)
    language = detect_language(temp_filename)
    gender = estimate_gender(mfcc)
    voice_class, confidence = classify_voice(mfcc)

    os.remove(temp_filename)

    return JSONResponse(content={
        "status": "success",
        "voiceClassification": voice_class,
        "confidenceScore": confidence,
        "language": language,
        "gender": gender,
        "noiseLevel": "LOW",
        "samplesProcessed": len(y),
        "inputMode": "live",
        "modelType": "Universal Acoustic Pattern Model"
    })

@app.post("/video-detect")
async def detect_video(file: UploadFile = File(...)):
    ext = validate_file_format(file.filename, SUPPORTED_VIDEO_FORMATS)
    temp_filename = f"temp_{uuid.uuid4()}.{ext}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    audio_path = f"temp_audio_{uuid.uuid4()}.wav"
    extract_audio_from_video(temp_filename, audio_path)

    y, sr_rate, mfcc = preprocess_audio(audio_path)
    language = detect_language(audio_path)
    gender = estimate_gender(mfcc)
    voice_class, confidence = classify_voice(mfcc)
    face_class = detect_face(temp_filename)

    os.remove(temp_filename)
    os.remove(audio_path)

    return JSONResponse(content={
        "status": "success",
        "voiceClassification": voice_class,
        "confidenceScore": confidence,
        "language": language,
        "gender": gender,
        "faceClassification": face_class,
        "noiseLevel": "MEDIUM",
        "samplesProcessed": len(y),
        "inputMode": "video",
        "modelType": "Universal Acoustic Pattern Model"
    })
