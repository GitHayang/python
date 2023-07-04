#!/usr/bin/env python
# coding: utf-8

# In[3]:


### 백엔드 서버 구성 라이브러리
from flask_cors import CORS
from flask import Flask, request, json, jsonify
### 데이터 처리 및 로드 라이브러리
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sqlalchemy import types

### 머신러닝 라이브러리
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
picklemodel = 'tree_model.pickle'

app = Flask(__name__)

picklemodel = 'tree_model.pickle'

### 웹에서 데이터 req

@app.route("/mypageRecommendation", methods=['POST'])
def post(): 
    # 웹에서 데이터 받기
    params = request.get_json()
    # type(params) : 리스트    # print(params[0]) #->{'GENDER': '1', 'AGE': '2', 'FOREINERINDEX': '1', 'INCOME': '1', 'CUST_CATEGORY': '1'}    # print(params[0]['GENDER']) #->1
    print(params)
    with open("mariaData.pickle", "rb") as mariaData:
        mariaDbInfo = pickle.load(mariaData)

    mariaId = mariaDbInfo.get("mariaId")
    mariaPw = mariaDbInfo.get("mariaPw")
    mariaIp = mariaDbInfo.get("mariaIp")
    mariaPort = mariaDbInfo.get("mariaPort")
    mariaDbName = mariaDbInfo.get("mariaDbName")
    mariaEngine = create_engine('mysql+pymysql://{}:{}@{}:{}/{}'.format(mariaId, mariaPw, mariaIp, mariaPort, mariaDbName))
    
    targetDf = pd.DataFrame.from_dict(params, orient="index").T
    
    tableName = "rawData"
    objectColumns = list(targetDf.columns[targetDf.dtypes == 'object'])
    typeDict={}
    maxLen = 200
    for i in range(0, len(objectColumns)):
        typeDict[ objectColumns[i] ] = types.VARCHAR(maxLen)

    targetDf.to_sql(name=tableName, if_exists="append", con=mariaEngine, index=False)
        
    #def age(ageValue):
    #    int(ageValue)
    #    if ageValue <= 9:
    #        return 0
    #    elif ageValue <= 19:
    #        return 1
    #    elif ageValue <= 29:
    #        return 2
    #    elif ageValue <= 39:
    #        return 3
    #    elif ageValue <= 49:
    #        return 4
    #    elif ageValue <= 59:
    #        return 5
    #    elif ageValue <= 69:
    #        return 6
    #    elif ageValue <= 79:
    #        return 7
    #    elif ageValue <= 89:
    #        return 8
    #    elif ageValue <= 99:
    #        return 9
    #    else:
    #        return 10

    #나이 인덱스 반환
    #ageValue = int(params[0]['AGE'])

    #params[0]['AGE'] = age(ageValue)
    #카테고리 인덱스 반환 ( 함수 )     
    #def income(incomeValue):
    #    int(incomeValue)
    #    if incomeValue <= 35000000:
    #        return 1
    #    elif incomeValue <= 70000000:
    #        return 2
    #    elif incomeValue <= 105000000:
    #        return 3
    #    elif incomeValue <= 140000000:
    #        return 4
    #    else:
    #        return 5

    #카테고리 인덱스 반환
    #incomeValue = int( params[0]['INCOME'] )
    #params[0]['INCOME'] = income(incomeValue)

    # 카테고리 인덱스 반환 ( np.where )
    # params['INCOME'] =  np.where( incomeValue<=35000000, 1,
    #                            np.where(incomeValue<=70000000,2,
    #                                np.where(incomeValue<=105000000,3,
    #                                    np.where(incomeValue<=140000000,4,5))))
    # 데이터 타입변경    
    dataInput = [
        int(params['GENDER']),
        int(params['AGE']),
        int(params['FOREINERINDEX']),
        int(params['INCOME']),
        int(params['CUST_TYPE'])
        ]

    # 모델 실행
    with open(picklemodel, 'rb') as f:
        loadedModel = pickle.load(f)
        inData = pd.DataFrame([dataInput])
        predictValue = loadedModel.predict(inData)

    def forWebTeam(params):
        """""
        입력한 딕셔너리 타입 param 변수를 바탕으로 형변환을 수행하는 함수입니다.
        입력 변수 param 내에는 GENDER, AGE, FOREINERINDEX, INCOME, CUST_TYPE 변수의 문자열 치환 기능을 수행합니다.
        출력 변수 param : 입력 변수 param의 형변환이 완료된 후, 동일한 이름의 변수로 이를 출력합니다.
        """""

        #GENDER = 0 or 1. 0 : 남성 / 1 : 여성
        if(params["GENDER"] == "0"):
            params["GENDER"] = "남성"
        else:
            params["GENDER"] = "여성"

        # AGE : 10년 단위로 인원의 나이를 구분하고, 이를 변수로 저장
        if(params["AGE"] == 0):
            params["AGE"] = "10세 미만"
        elif(params["AGE"] == 1):
            params["AGE"] = "10대"
        elif(params["AGE"] == 2):
            params["AGE"] = "20대"
        elif(params["AGE"] == 3):
            params["AGE"] = "30대"
        elif(params["AGE"] == 4):
            params["AGE"] = "40대"
        elif(params["AGE"] == 5):
            params["AGE"] = "50대"
        elif(params["AGE"] == 6):
            params["AGE"] = "60대"
        elif(params["AGE"] == 7):
            params["AGE"] = "70대"
        elif(params["AGE"] == 8):
            params["AGE"] = "80대"
        elif(params["AGE"] == 9):
            params["AGE"] = "90대"
        else:
            params["AGE"] = "100세이상"

        # 대상 인원의 외국인 여부 판별. 0 : 내국인 / 1 : 외국인
        if(params["FOREINERINDEX"] == "0"):
            params["FOREINERINDEX"] = "내국인"
        else:
            params["FOREINERINDEX"] = "외국인"

        # 대상 인원의 소득 분위 판별. 35000000 단위 구분이 이뤄진다.
        if(params["INCOME"] == 1):
            params["INCOME"] = "35,000,000미만"
        elif(params["INCOME"] == 2):
            params["INCOME"] = "70,000,000미만"
        elif(params["INCOME"] == 3):
            params["INCOME"] = "105,000,000미만"
        elif(params["INCOME"] == 4):
            params["INCOME"] = "140,000,000미만"
        else:
            params["INCOME"] = "140,000,000초과"

        # 대상 인원의 소득 발생 여부를 판단. 0 : 무소득 / 1 : 소득 발생 인원
        if(params["CUST_TYPE"] == "0"):
            params["CUST_TYPE"] = "무소득자"
        else:
            params["CUST_TYPE"] = "소득발생인원"
        return params

    forWebTeam(params)
    
    totalDf = pd.DataFrame.from_dict(params, orient="index").T
    totalDf["PREDICT_VALUE"] = predictValue

    tableName = "testDataWeb"
    objectColumns = list(totalDf.columns[totalDf.dtypes == 'object'])
    typeDict={}
    maxLen = 200
    for i in range(0, len(objectColumns)):
        typeDict[ objectColumns[i] ] = types.VARCHAR(maxLen)

    totalDf.to_sql(name=tableName, if_exists="append", con=mariaEngine, index=False)

    #웹으로 res
    response = {
        "result": "{}".format(predictValue)
    }    
    return jsonify(response)

### 파일이 실행하려면 main 함수(이프로그램이 처음 호출 되었을때 실행되는 코드)가 필요함, 노드의 app.js랑 같음
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=11088)

