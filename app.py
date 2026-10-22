
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)

messages = [
    "Win a free prize now",
    "Congratulations you won a lottery",
    "Claim your free gift today",
    "Click here to win money",
    "Limited offer buy now",
    "You have won a cash reward",
    "Free entry in a prize draw",
    "Urgent claim your reward",
    "Let's meet for lunch today",
    "Can we study together tomorrow",
    "Please send me the class notes",
    "Your assignment is due tomorrow",
    "Are you coming to college",
    "The meeting starts at ten",
    "I will call you later",
    "Please share the project file"
]

labels = [
    "spam", "spam", "spam", "spam",
    "spam", "spam", "spam", "spam",
    "not spam", "not spam", "not spam", "not spam",
    "not spam", "not spam", "not spam", "not spam"
]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(messages, labels)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            prediction = model.predict([message])[0]

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
