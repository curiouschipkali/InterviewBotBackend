from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template, make_response
from flask_cors import CORS
import io
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from openai import OpenAI
import warnings
import boto3
import uuid

load_dotenv()

frontend_uri = os.getenv("frontend_uri")
print(frontend_uri)
app = Flask(__name__)
CORS(app,  resources={
    r"/*": {"origins": [frontend_uri]}
})

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="Due to a bug, this method doesn't actually stream the response content",
)

client = OpenAI(api_key=os.getenv("API_KEY"))
uri = os.getenv("uri")
mongoclient = MongoClient(uri)
db = mongoclient["Chats"]
chat_history = db["History"]

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
print(AWS_ACCESS_KEY)

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = frontend_uri or "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    return '', 204

@app.route("/", methods=["GET", "POST"])
def hello_world():
    return "Hello, world!"

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        file = request.files["audio"]

        audio_file = io.BytesIO(file.read())
        audio_file.name = "audio.wav"
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

        transcription_text = transcription.text
        print("Transcription:", transcription_text)

        user_message = {"role": "user", "content": transcription_text}
        chat_history.insert_one(user_message)

        prvs_chat = list(chat_history.find())
        for chat in prvs_chat:
            chat.pop("_id", None)

        ai_response = generate_response(transcription_text, prvs_chat)
        print("AI Response:", ai_response)

        assistant_message = {"role": "assistant", "content": ai_response}
        chat_history.insert_one(assistant_message)

        s3_file_url = text_to_speech_and_upload(ai_response)

        return jsonify({
            "transcription": transcription_text,
            "ai_response": ai_response,
            "audio_file_url": s3_file_url
        })
    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

def generate_response(transcription_text, prvs_chats):
    prompt = """
You are an empathetic interview assistant. Your role is to conduct an interview with a user and gather information on the challenges they face regarding menstrual health and hygiene.

The challenges you need to focus on are:

Menstrual Hygiene and Sanitary Products
Access to Medical Support and Guidance
Pain Management and Symptoms
Emotional and Mental Well-being
Social Stigma and Awareness
Always begin by introducing yourself as an interview assistant, then ask about each of these challenges one by one. For each challenge, ask a maximum of 5-8 questions to understand their feelings and issues. If the user discusses any challenge not on the list but is genuine, evaluate it in the same way.

If the input is unrelated, gently tell the user their prompt was irrelevant and guide them back to the challenges. Respond empathetically to match the user's emotions. If they joke, reply with humor. If distressed, calm them and continue.

End the interview with 'thank you' if the user confirms they are done.
"""
    messages = [
        {"role": "system", "content": "You are a helpful interviewing assistant."},
        {"role": "user", "content": prompt},
    ] + prvs_chats

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
        assistant_response = completion.choices[0].message.content
        return assistant_response
    except Exception as e:
        raise RuntimeError(f"Error generating response: {str(e)}")

def text_to_speech_and_upload(message):
    try:
        file_key = f"response_{uuid.uuid4()}.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=message,
        )
        audio_stream = io.BytesIO()
        for chunk in response.iter_bytes():
            audio_stream.write(chunk)
        audio_stream.seek(0)

        s3_client.upload_fileobj(audio_stream, AWS_BUCKET_NAME, file_key)
        s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{file_key}"
        return s3_url
    except Exception as e:
        raise RuntimeError(f"Error uploading to S3: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)