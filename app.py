#installer les packages necessaires
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialiser l'application Flask
app = Flask(__name__)

# Charger les variables d'environnement
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialiser Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "projet"
embedding_dimension = 768
index = pc.Index(index_name)

# Charger le modèle et le scaler enregistrés
model_filename = 'models/gbm_model.pkl'
scaler_filename = 'models/scaler.pkl'
gbm_model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Initialiser le modèle SentenceTransformer
model_name = "all-mpnet-base-v2"
embedding_model = SentenceTransformer(model_name)

# Initialiser Google Cloud API Key pour Gemini 1.5 flash
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1)

# Mappings pour les catégories
espece_mapping = {
    "chien": 1,
    "chat": 0
}

race_mapping = {
    "Abyssin": 0,
    "Bengal": 1,
    "Berger Allemand": 2,
    "Bouledogue Français": 3,
    "British Shorthair": 4,
    "Dalmatien": 5,
    "Golden Retriever": 6,
    "Husky": 7,
    "Maine Coon": 8,
    "Persan": 9,
    "Pinscher nain": 10,
    "Siamois": 11,
    "Sphynx": 12,
    "Teckel": 13,
    "Westie": 14,
    "Yorkshire": 15
}

# Mappage pour la qualité du sommeil
sleep_score_mapping = {
    (7, 10): "bonne qualité de sommeil",
    (3, 6): "mauvaise qualité de sommeil",
    (0, 2): "pas de sommeil"
}

# Fonction pour prédire et retourner le résultat sous forme de JSON après le mapping
def predict_and_return_json(json_input):
    # Charger les données JSON
    data_dict = json.loads(json_input)

    # Convertir les noms en valeurs numériques
    data_dict['Espèce'] = espece_mapping.get(data_dict['Espèce'], -1)
    data_dict['Race'] = race_mapping.get(data_dict['Race'], -1)

    # Convertir le dictionnaire en DataFrame
    sample = pd.DataFrame([data_dict])

    # Normaliser l'échantillon avec le scaler
    sample_scaled = scaler.transform(sample)

    # Prédire la maladie pour l'échantillon
    prediction = gbm_model.predict(sample_scaled)[0]

    # Appliquer le mappage de score de sommeil
    score_sommeil = sample.iloc[0]['Score_sommeil']
    sleep_quality = "Inconnu"
    for (low, high), description in sleep_score_mapping.items():
        if low <= score_sommeil <= high:
            sleep_quality = description
            break

    # Créer des mappings pour les features et la prédiction
    espece_mapping_reverse = {v: k for k, v in espece_mapping.items()}
    race_mapping_reverse = {v: k for k, v in race_mapping.items()}
    intensite_activite_mapping = {0: 'Faible', 1: 'Moyenne', 2: 'Élevée'}
    maladie_mapping = {
        0: "cancer",
        1: "maladie cardiaque",
        2: "maladie endocriniennes",
        3: "maladies infectieuses",
        4: "maladies respiratoires",
        5: "sain"
    }

    # Appliquer les mappings sur les données
    mapped_data = {
        "Espèce": espece_mapping_reverse.get(sample.iloc[0]['Espèce'], "Inconnu"),
        "Âge": sample.iloc[0]['Âge'],
        "Poids": sample.iloc[0]['Poids'],
        "Race": race_mapping_reverse.get(sample.iloc[0]['Race'], "Inconnu"),
        "Température": sample.iloc[0]['Température'],
        "Respiration": sample.iloc[0]['Respiration'],
        "Pulse": sample.iloc[0]['Pulse'],
        "Intensité_activité": intensite_activite_mapping.get(sample.iloc[0]['Intensité_activité'], "Inconnu"),
        "Score_sommeil": score_sommeil,
        "Qualité_sommeil": sleep_quality,
        "Prédiction": maladie_mapping.get(prediction, "Inconnue")
    }

    return mapped_data

# Fonction pour générer une question basée sur le résultat de la prédiction
def generate_question(result_json):
    question = (
        f"Voici les données de l'animal :\n"
        f"- Température : {result_json['Température']}°C\n"
        f"- Rythme cardiaque : {result_json['Pulse']} BPM\n"
        f"- Rythme respiratoire : {result_json['Respiration']} respirations par minute\n"
        f"- Intensité d'activité : {result_json['Intensité_activité']}\n"
        f"- Score de sommeil : {result_json['Score_sommeil']} ({result_json['Qualité_sommeil']})\n\n"
        f"Le modèle de prédiction indique que l'animal pourrait souffrir de : {result_json['Prédiction']}.\n"
        f"Cet animal est un {result_json['Espèce']} de {result_json['Âge']} ans, de race {result_json['Race']}.\n\n"
        f"En tant que vétérinaire, veuillez analyser ces informations et fournir des recommandations appropriées. "
        f"Si l'animal est sain, fournissez des conseils généraux pour la prévention des maladies. "
        f"Si une maladie est détectée, fournissez des recommandations pour son traitement, y compris les vaccins recommandés, "
        f"les types d'alimentation conseillés, et comment maintenir un environnement propre et sain pour l'animal. "
        f"Décrivez également comment établir un plan de vaccination efficace pour cette espèce."
    )
    return question

# Fonction pour obtenir une réponse en utilisant LangChain et Pinecone
def get_answer(query):
    # Search for relevant chunks in Pinecone
    results = index.query(
        vector=embedding_model.encode(query).tolist(),
        top_k=10,
        include_metadata=True
    )
    # Get the texts of the relevant chunks
    texts = [result['metadata']['text'] for result in results['matches']]
    # Generate an answer based on the retrieved texts
    answer = llm.invoke(f"Answer this question based on the following context: {texts}\n\nQuestion: {query}")
    return answer

@app.route('/predict_and_answer', methods=['POST'])
def predict_and_answer():
    # Recevoir les données JSON
    input_data = request.json
    
    # Effectuer la prédiction
    result_json = predict_and_return_json(json.dumps(input_data))
    
    # Générer la question à poser
    question = generate_question(result_json)
    
    # Obtenir la réponse en utilisant Pinecone et LangChain
    answer = get_answer(question)
    
    # Combiner le résultat de la prédiction et la réponse
    response = {
        "resultat_prediction": {
            "Espèce": result_json["Espèce"],
            "Âge": result_json["Âge"],
            "Poids": result_json["Poids"],
            "Race": result_json["Race"],
            "Température": result_json["Température"],
            "Respiration": result_json["Respiration"],
            "Pulse": result_json["Pulse"],
            "Intensité d'activité": result_json["Intensité_activité"],
            "Score de sommeil": result_json["Score_sommeil"],
            "Qualité de sommeil": result_json["Qualité_sommeil"],
            "Prédiction": result_json["Prédiction"]
        },
        "question_veterinaire": question,
        "reponse": answer.content
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
