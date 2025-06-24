import pandas as pd
import numpy as np
import random

boy_names = [
    'Rohit', 'Swapnil', 'Sanket', 'Prathamesh', 'Omkar', 'Yash', 'Shubham', 'Akash',
    'Vaibhav', 'Tejas', 'Kiran', 'Gaurav', 'Saurabh', 'Amol', 'Bhushan', 'Sagar',
    'Sahil', 'Amit', 'Swanand', 'Chaitanya'
]
girl_names = [
    'Anjali', 'Sanjana', 'Siddhi', 'Triveni', 'Aishwarya', 'Snehal', 'Pooja', 'Prachi',
    'Shruti', 'Rutuja', 'Komal', 'Mayuri', 'Neha', 'Priya', 'Tanvi', 'Manasi', 'Nikita',
    'Rutu', 'Sayali', 'Mrunal'
]
names = boy_names + girl_names
activities = ['Yes', 'No']
areas = ['Rural', 'Urban']
performances = ['Good', 'Average', 'Poor']

rows = []
for i in range(5000):
    name = random.choice(names)
    if name in boy_names:
        gender = 'Male'
    else:
        gender = 'Female'
    age = random.randint(16, 20)
    extra = random.choice(activities)
    area = random.choice(areas)
    tenth = random.randint(60, 100)
    twelfth = random.randint(60, 100)
    study_hours = round(np.clip(np.random.normal(3, 1.2), 0.5, 8), 1)
    attendance = random.randint(40, 100)
    prev_cgpa = round(np.clip(np.random.normal(7, 1.5), 4, 10), 1)
    # Performance based on CGPA and study hours
    if prev_cgpa >= 8 and study_hours >= 4:
        perf = 'Good'
    elif prev_cgpa < 6 or study_hours < 2:
        perf = 'Poor'
    else:
        perf = 'Average'
    rows.append([
        name, age, gender, area, extra, tenth, twelfth, study_hours, attendance, prev_cgpa, perf
    ])

cols = [
    'Name', 'Age', 'Gender', 'AreaOfLiving', 'ExtraCurricular', 'TenthMarks', 'TwelfthMarks',
    'StudyHours', 'Attendance', 'PreviousCGPA', 'Performance'
]
df = pd.DataFrame(rows, columns=cols)
df.to_csv('student.csv', index=False) 