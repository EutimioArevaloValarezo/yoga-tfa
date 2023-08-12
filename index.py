from flask import *
import tensorflow as tf
from keras.models import load_model
from keras.applications.densenet import preprocess_input
from PIL import Image
import numpy as np
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="timpal0l/mdeberta-v3-base-squad2")
ruta_json_posiciones = "./posiciones.json"

with open(ruta_json_posiciones, 'r', encoding='utf-8') as file:
    datos = json.load(file)

def buscar_por_nombre(nombre):
    posiciones = datos['posiciones']
    for posicion in posiciones:
        if posicion['nombre'] == nombre:
            return posicion
    return None

def cargar_modelo():
    model = load_model('./models/densenet121_yoga_v1.h5')
    return model

app = Flask(__name__)

@app.route('/')
def home():
    try:
        return render_template('home.html')
    except Exception as e:
        return render_template('home.html')

@app.route('/informacion/Trikonasana')
def informacion():
    data = buscar_por_nombre('TRIKONASANA')
    return render_template('informacion.html', data = data, postura = "Trikonasana")

@app.route('/informacion/Utkata-Konasana')
def informacion2():
    data = buscar_por_nombre('UTKATA KONASANA')
    return render_template('informacion.html', data = data, postura = "Utkata_Konasana")

@app.route('/informacion/Virabhadrasana')
def informacion3():
    data = buscar_por_nombre('VIRABHADRASANA')
    return render_template('informacion.html', data = data, postura = "Virabhadrasana")

@app.route('/informacion/Vrikshasana')
def informacion4():
    data = buscar_por_nombre('VRIKSHASANA')
    return render_template('informacion.html', data = data, postura = "Vrikshasana")

@app.route('/preguntar', methods=['POST'])
def preguntar():
    pregunta = request.form['pregunta']
    postura = request.form['postura']
    data = buscar_por_nombre(str(postura))
    respuesta = qa_pipeline(question=pregunta, context=data['contexto'])
    respuesta = respuesta['answer']
    return jsonify({'respuesta': respuesta})


@app.route('/practicar')
def practicar():
    return render_template('practicar.html')

@app.route('/practicar/Trikonasana')
def practicar_Trikonasana():
    data = buscar_por_nombre('TRIKONASANA')
    return render_template('practicar.html', data = data)

@app.route('/practicar/Utkata-Konasana')
def practicar_Utkata_Konasana():
    data = buscar_por_nombre('UTKATA KONASANA')
    return render_template('practicar.html', data = data)

@app.route('/practicar/Virabhadrasana')
def practicar_Virabhadrasana():
    data = buscar_por_nombre('VIRABHADRASANA')
    return render_template('practicar.html', data = data)

@app.route('/practicar/Vrikshasana')
def practicar_Vrikshasana():
    data = buscar_por_nombre('VRIKSHASANA')
    return render_template('practicar.html', data = data)

if __name__ == '__main__':
    app.run(
        debug=True,
        extra_files=['./images/']
    )


