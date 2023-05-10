
import os
import openai
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from model1 import PoetryGenerator

app = Flask(__name__)
bootstrap = Bootstrap(app)

# Load the PoetryGenerator model
pg = PoetryGenerator('poem.txt')

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for the generate_poetry function
@app.route('/', methods=['GET', 'POST'])
def get_ai_poem():
    if request.method == 'POST':
        mood = request.form['user_input']
        pg.train_model(epochs=100)
        poem = pg.generate_poetry(mood)
        openai.api_key = os.getenv('API_TOKEN')
        response = jsonify(poem)
        return response
    return render_template('index.html')

# Define the route for the homepage
# @app.route('/', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         mood = request.form['user_input']
#         pg.train_model(epochs=1)
#         poem = pg.generate_poetry(mood)
#         openai.api_key = os.getenv('API_TOKEN')
#         response = jsonify(poem)
#         return render_template('index.html', ai_answer=response)
#     return render_template('index.html')

# Run app
if __name__ == '__main__':
    app.run(debug=True)