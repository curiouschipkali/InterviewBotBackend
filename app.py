from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
from pymongo import MongoClient
import config
from openai import OpenAI
import warnings

app = Flask(__name__)
CORS(app, supports_credentials=True)

warnings.filterwarnings("ignore", category=DeprecationWarning, 
                        message="Due to a bug, this method doesn't actually stream the response content, `.with_streaming_response.method()` should be used instead")

client = OpenAI(api_key=config.API_KEY)
uri = config.uri
mongoclient = MongoClient(uri)
db = mongoclient["Chats"]
chat_history = db['History']

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        file = request.files['audio']

        audio_file = io.BytesIO(file.read())
        audio_file.name = "audio.wav"
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

        transcription_text = transcription.text
        print(transcription_text)

        
        user_message = {
            "role": "user",
            "content": transcription_text
        }
        chat_history.insert_one(user_message)

        
        prvs_chat = list(chat_history.find())
        for chat in prvs_chat:
            chat.pop('_id', None) 

        
        ai_response = generate_response(transcription_text, prvs_chat)
        print("Generated AI Response:", ai_response)

        
        assistant_message = {
            "role": "assistant",
            "content": ai_response
        }
        chat_history.insert_one(assistant_message)

        
        audio_file_response = text_to_speech(ai_response)

        return send_file(
            audio_file_response,
            mimetype='audio/mp3',
            as_attachment=True,
            download_name="response.mp3"
        )

    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500


@app.route('/response', methods=['POST'])
def generate_response(transcription_text, prvs_chats):
    
    prompt = """You are an empathetic interview assistant. Your role is to conduct an interview with a user and gather information on challenges a user faces in his college or university. 
    The challenges you need to focus on are
    1. Hygiene
    2. Faculty and Study Resources
    3. Transport and Parking
    4. Timings
    5. Infrastructure
    
    You must always negin by introducing yourself as an interview assistant, then ask about each of these challenges one by one. For each challenge, ask the user a maximum of 5-8 questions to understand their feelings and issues. If the user discusses any challenge not on the list but is genuine, evaluate it in the same way.
    In case the input of the user is unrelated on any note, gently telk the user that their prompt was irreleant and guide them back to the challenges
    
    Also respond and match the emotion of the user. If they make a joke reply with Haha or If they seem distressed calm them and continue with the interview.
    If the user switches from one challenge to another without fully addressing the previous one evaluate what the user is speaking about now then, gently steer them back to the previous challenge. Once all challenges have been discussed, say 'thank you' and end the interview.
    
    Youre responsed should be thoughtful and empathetic, helping the user feel comfortable sharing. Keep your responses around 30 to 50 words.
    If the user at any point says that he is done with the interview, confirm whether to end the interview and if they confirm it
    again reply with the phrase 'thank you' and just this phrase with no extra text or nothing more .
    When evaluating a response, make sure not to ask more than 2 questions at once. Ask a question and a subsequent one not more than 2 at a time.
    """
    
    if not transcription_text or not transcription_text.strip():
        return jsonify({"error": "Transcription text is empty or invalid"}), 400

    messages = [
        {"role": "system", "content": "You are a helpful Interviewing assistant."},
        {"role": "user", "content": prompt},
    ] + prvs_chats

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        assistant_response = completion.choices[0].message.content
        return assistant_response
    except Exception as e:
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500

@app.route('/tts', methods=['POST'])
def text_to_speech(message):
    """
    Convert AI-generated text to speech using OpenAI API.
    """
    try:
        output_filename = "response.mp3"

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=message
        )
        response.stream_to_file(output_filename)
        return output_filename
    except Exception as e:
        raise RuntimeError(f"Error processing text-to-speech: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
