from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load pre-trained model and vectorizer
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = ""
    if request.method == "POST":
        user_text = request.form["tweet"]
        user_vec = vectorizer.transform([user_text])
        sentiment = model.predict(user_vec)[0]
    return render_template("index.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
