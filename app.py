from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import io
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from openai import OpenAI
import warnings

load_dotenv()

frontend_uri = os.getenv("frontend_uri")
print(frontend_uri)
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": frontend_uri}})

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

        audio_file_response = text_to_speech(ai_response)

        return send_file(
            audio_file_response,
            mimetype="audio/mp3",
            as_attachment=True,
            download_name="response.mp3",
        )

    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500


def generate_response(transcription_text, prvs_chats):
    prompt = """You are an empathetic interview assistant. Your role is to conduct an interview with a user and gather information on challenges a user faces in their college or university. 
    The challenges you need to focus on are:
    1. Hygiene
    2. Faculty and Study Resources
    3. Transport and Parking
    4. Timings
    5. Infrastructure

    Always begin by introducing yourself as an interview assistant, then ask about each of these challenges one by one. For each challenge, ask a maximum of 5-8 questions to understand their feelings and issues. If the user discusses any challenge not on the list but is genuine, evaluate it in the same way.

    If the input is unrelated, gently tell the user their prompt was irrelevant and guide them back to the challenges. Respond empathetically to match the user's emotions. If they joke, reply with humor. If distressed, calm them and continue.

    End the interview with 'thank you' if the user confirms they are done."""
    
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


def text_to_speech(message):
    """
    Convert AI-generated text to speech using OpenAI API.
    """
    try:
        output_filename = "response.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=message,
        )
        response.stream_to_file(output_filename)
        return output_filename
    except Exception as e:
        raise RuntimeError(f"Error processing text-to-speech: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
