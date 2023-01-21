from flask import Flask, render_template, request
import pickle
import numpy as np
from werkzeug.datastructures import ImmutableMultiDict

import json
import pandas as pd


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.json
        res = result
        arr1 = list(res.values())
        d = {"Not Interested": 1, "Poor": 2, "Beginner": 3, "Average": 5, "Intermediate": 6, "Excellent": 7,
             "Professional": 9}
        for i in range(len(arr1)):
            arr1[i] = d[arr1[i]]

        arr = ([value for value in arr1])

        data = np.array(arr)

        data = data.reshape(1, -1)

        loaded_model = pickle.load(open("careerlast.pkl", 'rb'))
        predictions = loaded_model.predict(data)

        pred = loaded_model.predict_proba(data)

        jobs_dict = {0: 'AI ML Specialist',
                     1: 'API Integration Specialist',
                     2: 'Application Support Engineer',
                     3: 'Business Analyst',
                     4: 'Customer Service Executive',
                     5: 'Cyber Security Specialist',
                     6: 'Data Scientist',
                     7: 'Database Administrator',
                     8: 'Graphics Designer',
                     9: 'Hardware Engineer',
                     10: 'Helpdesk Engineer',
                     11: 'Information Security Specialist',
                     12: 'Networking Engineer',
                     13: 'Project Manager',
                     14: 'Software Developer',
                     15: 'Software Tester',
                     16: 'Technical Writer'}
        result = {}
        pred = pred > 0.05
        pred = pred.tolist()[0]
        j = 0
        for i in range(len(pred)):
            if pred[i]:
                result[j] = jobs_dict[i]
                j += 1
        ans = json.dumps(result, indent = 4)

        return ans
            #render_template("testafter.html", final_res=final_res, job_dict=jobs_dict, job0=data1)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=50162)
