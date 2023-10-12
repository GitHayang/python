from flask_cors import CORS
from flask import Flask, request, json, jsonify
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib
import base64
import io

matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["axes.unicode_minus"] = False

app = Flask(__name__)
picklemodel = './hyulApDang.pickle'
app.config['JSON_AS_ASCII'] = False


with open(picklemodel, 'rb') as f:
    loaded_model = pickle.load(f)

CORS(app)

def generate_chart(status_counts, resultData):
    status_percentages = (status_counts / len(resultData)) * 100
    explode = [0.05]*4
    plt.pie(status_percentages, labels=status_counts.index, autopct='%1.1f%%', explode=explode)
    plt.title('Health Status')

    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    image_path = base64.b64encode(image_stream.getvalue()).decode('utf-8')

    return image_path

@app.route("/", methods=['POST'])
def uploaded_file():
    uploaded_file = request.files['file']
    
    if uploaded_file != '':
        try:
            df = pd.read_csv(uploaded_file)
        except:
            print("지원하는 형식의 파일이 아닙니다.")
    
    try:
        eightData = df.loc[mergedData['when_eat'] == "공복 (8시간 이상)"]
        eightData = df.reset_index(drop=True)
    except:
        eightData = df.copy()  
        eightData = df.reset_index(drop=True)
    
    try:
        learnData = eightData[['systolic', 'glucosedata']]
        learnData = learnData.reset_index(drop=True)
    except:
        try: 
            learnData = eightData[['systolic']]
            learnData['glucosedata'] = 0
            learnData = learnData.reset_index(drop=True)
        except:
            learnData = eightData[['glucosedata']]
            learnData['systolic'] = 0        
            learnData = learnData.reset_index(drop=True)

    status = []
    for i in range(0,len(learnData)):
        pred = pd.DataFrame(learnData.iloc[i])
        transposed_df = pred.transpose()
        transposed_df = transposed_df.rename(columns={"systolic":"SBP","glucosedata":"FBS"})
        status.append(loaded_model.predict(transposed_df))
    status = pd.DataFrame(status, columns=["status"])
    
    resultData = pd.concat([learnData, status], axis=1)
    resultData["result"] = ""
    resultData["result"] = np.where(resultData["status"] == 1, "hypertension + diabetes",
                                   np.where(resultData["status"] == 2, "hypertension", 
                                           np.where(resultData["status"] == 3, "diabetes", "health")))
    
    status_counts = resultData['result'].value_counts()

    total = (status_counts[0]+status_counts[1]+status_counts[2]+status_counts[3])
    diabetes = (status_counts[1]+status_counts[2])/total
    hypertension = (status_counts[1]+status_counts[3])/total
    
    image_path = generate_chart(status_counts, resultData)

    result = {}

    if diabetes >= 0.3 or hypertension >= 0.3:
        if diabetes >= 0.3 and diabetes < 0.4:
            result["diabetes"] = "당뇨 주의"
        elif diabetes >= 0.4:
            result["diabetes"] = "당뇨 위험"

        if hypertension >= 0.3 and hypertension < 0.4:
            result["hypertension"] = "고혈압 주의"
        elif hypertension >= 0.4:
            result["hypertension"] = "고혈압 위험"
    else : 
        result["status"] = "건강합니다."
        
    result["image"] = image_path
    
    return json.dumps(result, ensure_ascii=False)
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=34463)
