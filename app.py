from flask import Flask , request , render_template
from function import vns_preprocessing
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import pickle


app = Flask(__name__, template_folder='template')
model = pickle.load(open('models/sentiment_lstm.pk1','rb'))

with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(json.load(f))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    raw_text = request.form.values()
    text = str(raw_text)
    text = vns_preprocessing(raw_text)
    sequences_test  = tokenizer.texts_to_sequences(text)
    sequences_padded = pad_sequences(sequences_test, maxlen=128,
                                     padding='post', truncating='post')
    prediction = model.predict(sequences_padded)
    output = np.argmax(prediction,axis=1)
    sentiment = ''
    if output == 0:
        sentiment += 'Tệ'
    elif output == 1:
        sentiment += 'Trung bình'
    else:
        sentiment += 'Tốt'
    
    return render_template('index.html',prediction_text =sentiment)


if __name__ == "__main__":
    app.run(debug = True)