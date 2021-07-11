import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
# def hello_world():
#     return render_template("gui.html")
#
#
# @app.route('/predict', methods=['POST','GET'])
# def predict():
#     int_features=[int(x) for x in request.form.values()]
#     final=[np.array(int_features)]
#     print(int_features)
#     print(final)
#     prediction=model.predict_proba(final)
#     output='{0:.{1}f}'.format(prediction[0][1], 2)
#
#     if output>str(0.5):
#         return render_template('gui.html', pred='It is a Bullying.\n Bullying deteced {}'.format(output),bhai="What To do?")
#     else:
#         return render_template('gui.html', pred='Non Bullying.\n Bullying not detected {}'.format(output),bhai="Text is Safe")
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST', 'GET'])
def predict():

    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
#
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#
#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)