from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os
import requests
from dotenv import load_dotenv
from pydub import AudioSegment
import torch
import torchvision
import cv2
import numpy as np

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv('SID')
TWILIO_AUTH_TOKEN = os.getenv('AUTH_TOKEN')
NGROK_URL = os.getenv('NGROK_URL')

app = Flask(__name__)

ALLOWED_AUDIO_TYPES = ['audio/aac', 'audio/mpeg', 'audio/ogg', 'audio/wav', 
                       'audio/mp4', 'audio/amr', 'audio/x-wav', 'audio/waptt']

@app.route('/')
def index():
    return 'Welcome to the WhatsApp Media Processor!'

@app.route('/whatsapp', methods=['GET', 'POST'])
def process_whatsapp_message():
    # As error handling we wrap the main logic inside a try-except block
    # to catch any exceptions that may occur
    try:
        if 'MediaUrl0' in request.values:
            media_url = request.values['MediaUrl0']
            media_type = request.values.get('MediaContentType0', '')
            account_sid = request.values.get('AccountSid', 'Unknown')

            if media_type in ALLOWED_AUDIO_TYPES:
                save_audio_to_hard_drive(media_url, account_sid)
                return "Audio file received and saved successfully."
            elif media_type.startswith('image/'):
                save_image_with_face_detection(media_url, account_sid)
                return "Image received and processed successfully."
            else:
                return f"Unsupported media type: {media_type}"
        else:
            return "No media file received."
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return "An error occurred while processing the request.", 500

def save_audio_to_hard_drive(media_url, account_sid):
    account_directory = os.path.join("audio_files", account_sid)
    if not os.path.exists(account_directory):
        os.makedirs(account_directory)

    response = requests.head(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    content_disposition = response.headers.get('Content-Disposition')

    if content_disposition:
        # Extract the filename from the Content-Disposition header
        filename = content_disposition.split('filename=')[1].strip('"')
    else:
        # If Content-Disposition header is not available, 
        # use the base name of the URL as the filename
        filename = os.path.basename(media_url)
    
    downloaded_file = download_file(media_url)
    
    # Load the audio file using pydub
    audio = AudioSegment.from_file(downloaded_file)
    # Convert the audio to WAV format with 16kHz sampling rate
    wav_audio = audio.set_frame_rate(16000)
    
    file_count = len(os.listdir(account_directory))
    new_filename = f"audio_message_{file_count}.wav"
    
    wav_audio.export(os.path.join(account_directory, new_filename), format="wav")
    os.remove(downloaded_file)

    print(f"Audio file saved successfully as {new_filename}.")
def save_image_with_face_detection(media_url, account_sid):
    account_directory = os.path.join("images", account_sid)
    os.makedirs(account_directory, exist_ok=True)

    response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    image_data = response.content

    # Load the pre-trained SSD300 model for face detection
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    model.eval()

    # Convert the image data to a numpy array
    image_np = np.frombuffer(image_data, np.uint8)
    # Decode the image using OpenCV
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # Convert the color space from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess the image by converting it to a tensor
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    input_tensor = transform(image_rgb)

    # Perform face detection using the loaded model
    with torch.no_grad():
        output = model([input_tensor])[0]

    # Extract the bounding boxes and confidence scores from the model output
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    
    # Confidence score for a face to be detected.
    # Increasing the score may potentially reduce false positives.
    threshold = 0.75 

    # Filter the detected faces based on the confidence threshold
    detected_faces = boxes[scores >= threshold]

    if len(detected_faces) > 0:
        filename = f"image_{len(os.listdir(account_directory))}.jpg"
        output_path = os.path.join(account_directory, filename)
        # Save the image with detected faces
        cv2.imwrite(output_path, image)
        print(f"Face(s) detected. Image saved as {output_path}")
    else:
        print("No faces detected in the image.")

def download_file(url):
    try:
        response = requests.get(url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        if response.status_code == 200:
            temp_filename = 'temp_file'
            with open(temp_filename, 'wb') as f:
                f.write(response.content)
            return temp_filename
        else:
            raise Exception(f"Failed to download file from {url}. Status code: {response.status_code}")
    except Exception as e:
        app.logger.error(f"Error downloading file from {url}: {str(e)}")
        raise

if __name__ == '__main__':
    app.run(debug=True)