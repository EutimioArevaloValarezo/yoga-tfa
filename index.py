# pip install flask
# pip install cv2
# pip install tensorflow
# pip install transformers

from flask import *
import base64
import cv2
from keras.models import load_model
from keras.applications.densenet import preprocess_input
import numpy as np
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="timpal0l/mdeberta-v3-base-squad2")
ruta_json_posiciones = "./posiciones.json"
posturas = ["TRIKONASANA", "UTKATA KONASANA", "VIRABHADRASANA", "VRIKSHASANA"]
model = load_model('./models/densenet121_yoga_v1.h5')

def encontrar_indice(lista, elemento):
    for i, valor in enumerate(lista):
        if valor == elemento:
            return i
    return -1 

def predecir(image_data, postura):

    indice = encontrar_indice(posturas, postura)
    print("Postura: ",postura," Indice:",indice)
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_array, flags=1)

    # Corregir el balance de blancos (convertir a escala de grises y luego de nuevo a RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_corrected = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Preprocesar la imagen para que coincida con los requisitos de DenseNet121
    img_resized = cv2.resize(img_corrected, (224, 224))
    img_resized = img_resized / 255.0
    img_processed = np.expand_dims(img_resized, axis=0)

    # Realizamos la predicción
    classes = model.predict(img_processed)
    presicion = round(classes[0][indice]*100, 2)

    return presicion

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

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.form['image_data']
    postura = request.form['postura']
    prediction = predecir(image_data, postura)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', 
            port=5000, 
            debug=True)


