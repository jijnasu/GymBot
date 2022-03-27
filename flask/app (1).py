
from flask import Flask, jsonify, request
import pickle
import numpy as np


model = pickle.load(open('kmean1.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "hello world";


@app.route('/predict', methods = ['POST'])
def predict():
    a = request.form.get('a');
    b = request.form.get('b');
    c = request.form.get('c');
    d = request.form.get('d');
    e = request.form.get('e');
    f = request.form.get('f');
    g = request.form.get('g');



    #features = [536365,"85123A","White metal lantern",6,"12/1/2001",3.39,17850]
    features = [a,b,c,d,e,f,g]


    input_query = np.array([features])

    result = model.predict(input_query)[0]


    #result = {'a':a, 'b':b, 'c':c}
    #return jsonify(result)
    return jsonify({'crisk':str(result)})

if __name__  == '__main__':
    app.run(debug=True)