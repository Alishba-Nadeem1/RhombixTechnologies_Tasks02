from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Tweets.csv")

# Select only sentiment and text
df = df[['text', 'airline_sentiment']]

# Features & labels
X = df['text']
y = df['airline_sentiment']

# Convert text to numeric
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

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
