## Step 1. Defining the Purpose of the Bot

- Save audio messages from the chat window to database (DBMS or hard drive) by user’s character id of the account the message is associated with
- Convert all audio messages in .wav format with 16khz frequency
- Save only those sent picture images that contain face. 

## Step 2. Identify the Type of Bot

There are three types of bots: rule-based, AI powered using ML algorithms and hybrid. Hybrid chatbots combine the best of both worlds. It will use predefined rules for storing and converting audio messages and machine learning for identifying face on the photo.

## Step 3. Select a channel through which  the Bot will communicate

WhatsApp will be selected as the channel. Other option included Telegram

## Step 4. Choose the Technology Stack

- **Twilio** is a cloud communications platform as a service (CPaaS) company which allows software developers to programmatically make and receive phone calls, send and receive text messages, and perform other communication functions using its web service APIs. With the Twilio API for WhatsApp, you can send notifications, have two-way conversations, or build chatbots. [Twilio](https://www.twilio.com/en-us?_gl=1*1wb6rqi*_ga*MTY0ODMzOTU5Ny4xNzEyMDYwNTQ5*_ga_8W5LR442LD*MTcxMjIzNjI2Mi4xLjAuMTcxMjIzNjI2Mi4wLjAuMA..) 

- **Flask** is a micro web framework written for Python. It’s lightweight, open source and offers a small and easily extensible core. It’s used primarily to develop minimalistic web applications and Rest APIs.

- **Ngrok** allows you to create secure ingress to any app, device or service without spending hours learning arcane networking technologies. [Ngrok](https://ngrok.com/)

- **pydub** is a Python library that provides easy manipulation of audio files.

- **PyTorch** offers pre-trained models for various computer vision tasks, including face detection. One popular face detection model in PyTorch is the SSD (Single Shot MultiBox Detector) model.

## Step 5. Design and test the Bot

#### Run virtual environment

```
python -m venv <env_name>
```

or 

```
virtualenv <env_name>
```

#### Install necessary libraries from requiremts.txt
```
pip install -r requirements.txt
```

#### Install Twilio and flask

```
pip install twilio flask
```

#### Set up a [Twilio](https://www.twilio.com/try-twilio) account

- Sign up for a Twilio account if you don’t have one.
- Obtain your Account SID and Auth Token from the Twilio dashboard and save them .env file
- Obtain the temporary telephone number for the testing purpose

#### Register and install [ngrok](https://dashboard.ngrok.com/get-started/setup/windows)

Follow instructions from ngrok [website](https://ngrok.com/download)
#### Create a new file **bot.py** and set up basic Flask application 

Creating Flask application that processes media files sent through WhatsApp. It should use Twilio API to receive and process the media files. The application checks if the received media file is an audio or an image and then processes it accordingly. 

For audio files it saves the file to a specific directory with a unique filename. It also converts the audio file to a WAV format with 16kHz sampling rate. 

For image file it uses a pre-trained SSD300 model for face detection. It loads the model, preprocesses the image and then performs face detection. If any faces are detected, it saves the image with the detected faces. 

The Application should also include error handling to catch any exceptions that may occur during the processing of the media files. 

#### Expose the Flask app using ngrok

```
ngrok http 5000
```

#### Configure Twilio Sanbox settings

- Copy endpoint URL from ngrok dashboard and paste it to Twilio Sanbox settings page adding **/whatsapp** at the end of URL (for example https://123456.ngrok-free.app/whatsapp) 
- Save configuration in Twilio
- Save copy of the NGROK_URL to .env file
- Note that ngrok URLs are dynamically generated and do not remain the same across sessions. The above steps need to be repeated with every new session. 

#### Run the Flask app

```
python bot.py
```

#### Test the bot 

- Send an audio file in WhatsApp to the number provided by Twilio
- Check the terminal for the bot response 
- Check if the file has been saved in audio_files folder
- Send a picture in WhatsApp to the same number containing a face
- Check the terminal for the bot response 
- Check if the picture has been saved in images folder
- Repeat the above steps for image without face
- If false positive images are saved to the folder, adjust threshold parameter in save_image_with_face_detection() function 
