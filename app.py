from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load or train model
def train_model():
    try:
        df = pd.read_csv('student.csv')
        # One-hot encode AreaOfLiving
        df = pd.get_dummies(df, columns=['AreaOfLiving'], drop_first=True)
        X = df[['StudyHours', 'Attendance', 'PreviousCGPA', 'AreaOfLiving_Urban']]
        y = df['Performance']
        model = DecisionTreeClassifier()
        model.fit(X, y)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        return model
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return None

if not os.path.exists('model.pkl'):
    model = train_model()
    if model is None:
        print("Failed to train model. Using default settings.")
        model = DecisionTreeClassifier()
else:
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = DecisionTreeClassifier()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        # AreaOfLiving from form (default to Rural if not present)
        area = data.get('area', 'Rural')
        area_urban = 1 if area == 'Urban' else 0
        features = [
            float(data['study_hours']),
            float(data['attendance']),
            float(data['cgpa']),
            area_urban
        ]
        prediction = model.predict([features])[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'Invalid input data'}), 400

@app.route('/visualize')
def visualize():
    try:
        df = pd.read_csv('student.csv')
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='StudyHours', y='PreviousCGPA', hue='Performance', data=df)
        plt.title('Study Hours vs CGPA')
        plt.savefig('static/graph.png')
        plt.close()
        return render_template('graph.html', img_path='static/graph.png')
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        return "Error generating visualization. Please check the data file.", 500

@app.route('/student_names')
def student_names():
    df = pd.read_csv('student.csv')
    names = df['Name'].tolist()
    return jsonify({'names': names})

@app.route('/all_students')
def all_students():
    df = pd.read_csv('student.csv')
    students = df.to_dict(orient='records')
    return jsonify({'students': students})

@app.route('/gender_performance')
def gender_performance():
    df = pd.read_csv('student.csv')
    # Group by Gender and Performance, then count
    perf_counts = df.groupby(['Gender', 'Performance']).size().unstack(fill_value=0)
    perf_counts = perf_counts[['Good', 'Average', 'Poor']] if set(['Good', 'Average', 'Poor']).issubset(perf_counts.columns) else perf_counts
    perf_counts.plot(kind='bar', figsize=(8,6))
    plt.title('Performance Comparison by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('static/gender_performance.png')
    plt.close()
    return render_template('gender_performance.html', img_path='static/gender_performance.png')

@app.route('/student_performance_graph/<student_name>')
def student_performance_graph(student_name):
    df = pd.read_csv('student.csv')
    student = df[df['Name'] == student_name].iloc[0]
    features = ['StudyHours', 'PreviousCGPA', 'Attendance']
    student_values = [student['StudyHours'], student['PreviousCGPA'], student['Attendance']]
    class_averages = [df[f].mean() for f in features]
    x = range(len(features))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar([i - width/2 for i in x], student_values, width, label='Student', color='#4e79a7')
    plt.bar([i + width/2 for i in x], class_averages, width, label='Class Average', color='#f28e2b')
    plt.xticks(x, features)
    plt.title(f"{student_name} - Performance: {student['Performance']}")
    plt.ylabel('Value')
    plt.ylim(0, max(max(student_values), max(class_averages), 10, 100))
    for i, v in enumerate(student_values):
        plt.text(i - width/2, v + 1, str(round(v, 2)), ha='center', color='#4e79a7')
    for i, v in enumerate(class_averages):
        plt.text(i + width/2, v + 1, str(round(v, 2)), ha='center', color='#f28e2b')
    plt.legend()
    plt.tight_layout()
    img_path = f'static/{student_name}_performance.png'
    plt.savefig(img_path)
    plt.close()
    return render_template('student_performance_graph.html', img_path=img_path, student_name=student_name, performance=student['Performance'])

@app.route('/overall_performance')
def overall_performance():
    df = pd.read_csv('student.csv')
    perf_counts = df['Performance'].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(perf_counts, labels=perf_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4e79a7','#f28e2b','#e15759'])
    plt.title('Overall Student Performance Distribution')
    plt.tight_layout()
    plt.savefig('static/overall_performance.png')
    plt.close()
    return render_template('overall_performance.html', img_path='static/overall_performance.png')

if __name__ == '__main__':
    app.run(debug=True)
