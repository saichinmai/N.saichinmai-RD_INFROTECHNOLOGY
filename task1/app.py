from flask import Flask, render_template, request, jsonify
import pickle
import json
import random
import re

app = Flask(__name__)

# Load trained model
model = pickle.load(open("chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("words.pkl", "rb"))

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

# Sample Order Database
orders = {
    "ORD101": {
        "status": "Shipped",
        "delivery": "Expected in 2 days"
    },
    "ORD102": {
        "status": "Out for Delivery",
        "delivery": "Arriving today"
    },
    "ORD103": {
        "status": "Delivered",
        "delivery": "Delivered successfully"
    },
    "ORD104": {
        "status": "Processing",
        "delivery": "Will be shipped soon"
    }
}


def get_response(message):

    message = message.lower()

    # Detect Order ID
    order_match = re.search(r'ord\d+', message)

    if order_match:
        order_id = order_match.group().upper()

        if order_id in orders:
            order = orders[order_id]

            return (
                f"Order ID: {order_id}\n"
                f"Status: {order['status']}\n"
                f"Delivery: {order['delivery']}"
            )
        else:
            return "Sorry, your order ID is not found."

    # Normal chatbot prediction
    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)[0]

    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():

    user_message = request.form["message"]

    response = get_response(user_message)

    return jsonify({"reply": response})


if __name__ == "__main__":
    app.run(debug=True)