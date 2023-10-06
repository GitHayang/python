from flask_cors import CORS
from flask import Flask, request, json, jsonify
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["axes.unicode_minus"] = False

app = Flask(__name__)
picklemodel = './hyulApDang.pickle'
app.config['JSON_AS_ASCII'] = False
# 모델 실행
with open(picklemodel, 'rb') as f:
    loaded_model = pickle.load(f)

CORS(app)

def generate_chart(status_counts, resultData):
    # 데이터 처리 및 그래프 생성
    status_percentages = (status_counts / len(resultData)) * 100
    explode = [0.05]*4
    plt.pie(status_percentages, labels=status_counts.index, autopct='%1.1f%%', explode=explode)
    plt.title('건강 상태')

    # 이미지를 저장할 경로 설정 (Flask 애플리케이션 경로 내에 있어야 함)
    image_path = 'C:/kopo/sts-4.19.1.RELEASE/workspace/tester/src/main/resources/static/generated_image.png'  # 경로 및 파일명은 상황에 맞게 수정

    # 이미지를 파일로 저장
    plt.savefig(image_path, format='png', encoding='utf-8')

    # 저장한 이미지 경로 반환
    return image_path

@app.route("/", methods=['POST'])
def uploaded_file():
    uploaded_file = request.files['file']
    
    if uploaded_file != '':
        try:
            df = pd.read_csv(uploaded_file)
            # 원본에서는 2개의 파일을 합쳤지만 받는건 1개만 받는다.
            # 1개의 파일안에 혈당, 혈압이 다 있을 수도 있고, 2중 1개만 있을 수도 있다.
        except:
            print("파일 못 받는중")
    
    # 혈당은 공복혈당을 기준으로 재지만, 공복혈당이 없거나,
    # 언제 쟀는지 표기가 없다면 그냥 혈당 전체를 사용한다.
    try:
        eightData = df.loc[mergedData['when_eat'] == "공복 (8시간 이상)"]
        eightData = df.reset_index(drop=True)
    except:
        eightData = df.copy()  
        eightData = df.reset_index(drop=True)
    
    # 파일에 혈압이나 혈당이 있을수도 무언가 빠져있을 수도 있기에 예외처리한다.
    # 없는 자료에는 0을 넣어 대처한다.
    try:
        learnData = eightData[['systolic', 'glucosedata']]
        learnData = learnData.reset_index(drop=True)
    except:
        try: # glucosedata가 없을때 
            learnData = eightData[['systolic']]
            learnData['glucosedata'] = 0
            learnData = learnData.reset_index(drop=True)
        except: # systolic이 없을때
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
    resultData["result"] = np.where(resultData["status"] == 1, "고혈압 + 당뇨",
                                   np.where(resultData["status"] == 2, "고혈압", 
                                           np.where(resultData["status"] == 3, "당뇨", "건강")))
    
    status_counts = resultData['result'].value_counts()

    total = (status_counts[0]+status_counts[1]+status_counts[2]+status_counts[3])
    diabetes = (status_counts[1]+status_counts[2])/total
    hypertension = (status_counts[1]+status_counts[3])/total                           

    # 이미지 생성
    image_path = generate_chart(status_counts, resultData)

    result = {}

    if diabetes >= 0.3 or hypertension >= 0.3:
        # 당뇨
        if diabetes >= 0.3 and diabetes < 0.4:
            result["diabetes"] = "당뇨 주의"
        elif diabetes >= 0.4:
            result["diabetes"] = "당뇨 위험"

        # 고혈압
        if hypertension >= 0.3 and hypertension < 0.4:
            result["hypertension"] = "고혈압 주의"
        elif hypertension >= 0.4:
            result["hypertension"] = "고혈압 위험"
    else : 
        result["status"] = "건강합니다."
        
    result["image_path"] = image_path

    return json.dumps(result, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=34463)
