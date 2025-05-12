from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Flask: classe que gerencia a aplicação
app = Flask(__name__)
CORS(app)

# Carregar o modelo
model = joblib.load("modelo_trem.pkl")

@app.route("/check_status")
def teste():
    return "Server is up!"

@app.route("/call_predict", methods=['POST'])
def chama_modelo():
    data = request.json['data']

    # Garantir que os dados estejam no formato correto
    data_array = np.array(data).reshape(1, -1)

    try:
        # Realizar a predição
        pred = model.predict(data_array)[0]
        return jsonify({"prediction": pred})
    except Exception as e:
        print(f"Erro durante a predição: {str(e)}")
        return jsonify({"error": f"Erro ao fazer a predição: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
