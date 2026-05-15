from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.form.get("Body")
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(f"You said: {incoming_msg}")  # simple echo test
    return str(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)