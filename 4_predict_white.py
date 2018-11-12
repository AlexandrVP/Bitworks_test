
# coding: utf-8

# In[4]:


from sklearn.externals.joblib import load
from sklearn.ensemble import BaggingClassifier

from flask import Flask
from flask_restful import Resource, Api
from flask_restful import reqparse


# In[15]:


app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def get(self):
        
        # Парсер для адресной браузерной строки
        parser = reqparse.RequestParser()
        parser.add_argument('fixed acidity', type=float, help=' cannot be converted')
        parser.add_argument('volatile acidity', type=float, help=' cannot be converted')
        parser.add_argument('citric acid', type=float, help=' cannot be converted')
        parser.add_argument('residual sugar', type=float, help=' cannot be converted')
        parser.add_argument('chlorides', type=float, help=' cannot be converted')
        parser.add_argument('free sulfur dioxide', type=float, help=' cannot be converted')
        parser.add_argument('total sulfur dioxide', type=float, help=' cannot be converted')
        parser.add_argument('density', type=float, help=' cannot be converted')
        parser.add_argument('pH', type=float, help=' cannot be converted')
        parser.add_argument('sulphates', type=float, help=' cannot be converted')
        parser.add_argument('alcohol', type=float, help=' cannot be converted')
        parser.add_argument('quality', type=float, help=' cannot be converted')
        args = parser.parse_args()
        
        # признаки за исключчением quality
        prediction = predict([[
                args['fixed acidity'], 
                args['volatile acidity'], 
                args['citric acid'], 
                args['residual sugar'],
                args['chlorides'], 
                args['free sulfur dioxide'], 
                args['total sulfur dioxide'],
                args['density'], 
                args['pH'], 
                args['sulphates'], 
                args['alcohol']
                ]])
        
        # Функция возвращает
        return {
#                 'fixed acidity': args['fixed acidity'], 
#                 'volatile acidity': args['volatile acidity'], 
#                 'citric acid': args['citric acid'], 
#                 'residual sugar': args['residual sugar'],
#                 'chlorides': args['chlorides'], 
#                 'free sulfur dioxide': args['free sulfur dioxide'], 
#                 'total sulfur dioxide': args['total sulfur dioxide'],
#                 'density': args['density'], 
#                 'pH': args['pH'], 
#                 'sulphates': args['sulphates'], 
#                 'alcohol':args['alcohol'],
                'Wine prediction quality':prediction,
                'Wine true quality':args['quality']
                  }

def predict(InputFeatures):
    
    # загружаем модель
    White_wine_predict_model = load('White_wine_quality_RF.sav')
    
    # переменная для записи классификации вина
    predictInt = White_wine_predict_model.predict(InputFeatures)
    
    # строка вывода в зависимости от оценки классификатора
    if predictInt[0] == 1:
        predictString = 'Bad Wine'
    elif predictInt[0] == 2:
        predictString = 'Average Wine'
    elif predictInt[0] == 3:
        predictString = 'Good Wine'
    else:
        predictString = 'null'
        
    return predictString

api.add_resource(Prediction, '/prediction')

if __name__ == '__main__':
    app.run(debug=False)


# In[3]:


# import numpy as np

# Features = np.asarray([7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4])
# Features = inputFeatures.reshape(1, -1)
# Predict =  predict(Features)

# print(Predict)

