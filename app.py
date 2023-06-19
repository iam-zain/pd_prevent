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
    gender = request.form.get('Gender', '')
    age = request.form.get('Age', '')

    num_features = int(request.form.get('num_features', '0'))

    feature_scores = [f'Feature{i}' for i in range(1, num_features + 1)]

    scores = {}
    for feature in feature_scores:
        score = request.form.get(f'Score_{feature}', '')
        scores[feature] = score

    if len(feature_scores) < 3:
        return render_template('error.html', message='Please select at least three features.')

    input_vec = []
    for feature in feature_scores:
        input_vec.append(request.form.get(feature))
        input_vec.append(scores[feature])
    input_str = ' '.join(input_vec)

    command = f'python ThreeModelPrediction.py {age} {len(feature_scores)} {input_str}'
    output = subprocess.getoutput(command)
    out_split = output.split("#")
    numeric_value = int(''.join(filter(str.isdigit, output)))

    data = {
        'output':output, 'numeric_value':numeric_value
    }
    return out_split

if __name__ == '__main__':
    app.run(debug=True)
