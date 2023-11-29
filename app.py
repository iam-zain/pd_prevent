from flask import Flask, render_template, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/box')
def box():
    return render_template('box.html')

@app.route('/submit', methods=['POST'])
def submit():
    age = request.form.get('Age', '')
    
    gender = request.form.get('Gender', '')
    
    num_features = int(request.form.get('num_features', '0'))

    feature_scores = [f'Feature{i}' for i in range(1, num_features + 1)]

    scores = {}
    for feature in feature_scores:
        score = request.form.get(f'Score_{feature}', '')
        scores[feature] = score

    if len(feature_scores) < 5:
        return render_template('error.html', message='Please select at least five features.')

    input_vec = []
    for feature in feature_scores:
        input_vec.append(request.form.get(feature))
        input_vec.append(scores[feature])
    input_str = ' '.join(input_vec)

    command = f'python ThreeModelPrediction.py {age} {gender} {len(feature_scores)} {input_str}'
    output = subprocess.getoutput(command)
    out_split = output.split("#")
    userInfo = out_split[0]

    if "Patient" in userInfo:
        userStatus = "PROFILE_1"
    elif "Healthy" in userInfo:
        userStatus = "PROFILE_2"
    else:
        userStatus = "Unknown"

    confidenceScore = int(out_split[1])
    
    data = {'userStatus':str(userStatus), 'confidenceScore': confidenceScore}
    return data
    

if __name__ == '__main__':
    app.run(debug=True)
