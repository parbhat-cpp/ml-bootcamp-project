from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=str(request.form.get('gender')),
            race_ethnicity=str(request.form.get('race')),
            lunch=str(request.form.get('lunch')),
            parental_level_of_education=request.form.get('education'),
            reading_score=float(str(request.form.get('reading'))),
            test_preparation_course=str(request.form.get('prep')),
            writing_score=float(str(request.form.get('writing'))),
        )
        pred_df = data.get_data_as_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html',result=results[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0')
