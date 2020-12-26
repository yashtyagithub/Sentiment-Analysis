from flask import Flask, render_template, request
import re
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model



app = Flask(__name__)



def init():
    global model
    model = load_model('Sentiment analysis model.h5')


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def Sentiment_analysis_prediction():
    if request.method=='POST':
        text = request.form['text']
        sentiment = ''
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text=re.sub(strip_special_chars, "", text.lower())

        words = text.split() #split string into a list
        x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=10000) else 0 for word in words]]
        x_test = sequence.pad_sequences(x_test, maxlen=500) # Should be same which you used for training data
        probability = model.predict(x_test)[0][0]
        probability=round(probability,3)
        class1 = model.predict_classes(x_test)
        if class1 == 0:
            sentiment = 'Negative'


        else:
            sentiment = 'Positive'

    return render_template('home.html', text=text, sentiment=sentiment, probability=probability)



if __name__ == "__main__":
    init()
    app.run(debug=True)
