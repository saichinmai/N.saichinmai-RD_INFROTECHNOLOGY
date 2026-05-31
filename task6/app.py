from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(
    open("news_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)


@app.route("/")
def home():
    return render_template(
        "index.html"
    )


@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    transformed_news = (
        vectorizer.transform(
            [news]
        )
    )

    prediction = model.predict(
        transformed_news
    )[0]

    return render_template(
        "index.html",
        prediction=prediction,
        news=news
    )


if __name__ == "__main__":
    app.run(debug=True)