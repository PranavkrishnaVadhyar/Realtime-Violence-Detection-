import time
from twilio.rest import Client  # Import Twilio Client
import json

# Load Twilio credentials from the JSON file
with open('credentials.json') as config_file:
    twilio_config = json.load(config_file)

TWILIO_ACCOUNT_SID = twilio_config['TWILIO_ACCOUNT_SID']
TWILIO_AUTH_TOKEN = twilio_config['TWILIO_AUTH_TOKEN']
TWILIO_PHONE_NUMBER = twilio_config['TWILIO_PHONE_NUMBER']
RECIPIENT_PHONE_NUMBER = twilio_config['RECIPIENT_PHONE_NUMBER']

# Initialize the Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize last SMS sent time
last_sms_sent_time = 0

def send_sms(timestamp):
    global last_sms_sent_time

    current_time = time.time()
    
    # Check if it's been more than 1 minute since the last SMS was sent
    if current_time - last_sms_sent_time >= 60:
        message_body = f"Violence detected at {timestamp}."

        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )

        print("SMS sent successfully.")

        # Update last SMS sent time
        last_sms_sent_time = current_time
    else:
        print("SMS not sent. Waiting for cooldown period.")
