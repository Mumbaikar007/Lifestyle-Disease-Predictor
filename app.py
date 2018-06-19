
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dataset - Sheet1.csv')
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC( kernel = 'linear', random_state = 0 )
classifier.fit(X, y)


from flask import Flask, render_template, request, jsonify, redirect, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')


@app.route('/api/get_prediction', methods=['POST'])
def say_name():

    json = request.get_json()


    unhealthyEatingHabits = int(json['unhealthyEatingHabits'])
    lackOfPhysicalActivity = int(json['lackOfPhysicalActivity'])
    stressAndAnxiety = int(json['stressAndAnxiety'])
    obesity = int(json['obesity'])
    alcoholism = int(json['alcoholism'])
    poorSleep = int(json['poorSleep'])
    gender = int(json['gender'])
    anotherLifestyle = int(json['anotherLifestyle'])
    historyLifetyle = int(json['historyLifetyle'])
    smoking = int(json['smoking'])
    age = int(json['age'])

    tp = [ unhealthyEatingHabits, lackOfPhysicalActivity, obesity,
           stressAndAnxiety, poorSleep, smoking, alcoholism, anotherLifestyle,
           historyLifetyle, gender, age]
    tp = np.array(tp)
    tp = tp.reshape(1, -1)

    result = classifier.predict(tp)[0]

    if result == 0:
        result = "You don't suffer from a lifestyle disease!"
        #return redirect("http://127.0.0.1:5000/templates/notSuffer.html", code=302)
        #return render_template('notSuffer.html')
        return jsonify (result=result)
        #return jsonify(unhealthyEatingHabits=unhealthyEatingHabits)

    result = "You suffer from a lifestyle disease! "
    return jsonify(result=result)


if __name__ == '__main__':
    app.run(debug=True)
