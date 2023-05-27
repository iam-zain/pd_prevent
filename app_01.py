from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    gender = request.form.get('Gender', '')
    age = request.form.get('Age', '')
    feature_scores = request.form.getlist('Feature')

    scores = {}
    for feature in feature_scores:
        score = request.form.get(f'Score_{feature}', '')
        scores[feature] = score

    if len(feature_scores) < 3:
        return render_template('error.html', message='Please select at least three features.')

    input_vec = []
    for feature in feature_scores:
        input_vec.append(feature)
        input_vec.append(scores[feature])
    input_str = ' '.join(input_vec)

    command = f'python ThreeModelPrediction.py {age} {len(feature_scores)} {input_str}'
    output = subprocess.getoutput(command)
    out_term = output.split()[7]
    numeric_value = float(''.join(filter(str.isdigit, output)))

    if out_term == 'Patient,':
        gauge_color = 'danger'
    elif out_term == 'Healthy,':
        gauge_color = 'success'
    else:
        gauge_color = 'warning'

    return render_template('result.html', output=output, gauge_color=gauge_color, numeric_value=numeric_value)

if __name__ == '__main__':
    app.run()
