# 🐾 Pets Health API

API de recommandation pour la **santé animale**, permettant de prédire des maladies chez les animaux de compagnie (chiens et chats) à partir de leurs signes vitaux et de fournir des recommandations adaptées.  

---

## 🚀 Fonctionnalités

- 🔍 **Prédiction de maladies animales** via modèles de **Machine Learning**  
- 💡 **Recommandations personnalisées** grâce à un système **RAG (Retrieval-Augmented Generative)** avec LLM (Gemini + LangChain)  
- 🌐 **API REST** développée avec **Flask**  
- 🐳 **Conteneurisation** complète avec **Docker** pour simplifier le déploiement  
- 🧪 **Tests API** via Postman  

---

## 🛠️ Technologies utilisées

- **Python 3.12**  
- **Flask** – création de l’API  
- **scikit-learn** – entraînement des modèles (SVM, Decision Tree, Random Forest, Gradient Boosting)  
- **SentenceTransformers** – génération d’embeddings pour la recherche sémantique  
- **Pinecone** – base de données vectorielle  
- **LangChain + Gemini** – génération de réponses enrichies  
- **Docker** – conteneurisation et déploiement  

---

## 📦 Installation locale

```bash
# 1. Cloner le projet
git clone https://github.com/Essra-Hmida/pets_health_API.git
cd pets_health_API

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l’API
python app.py

---

## 📡 Tester l’API avec Postman

```bash
# 🔗 Endpoint principal
POST http://localhost:5000/predict_and_answer


# Body (JSON) :

{
  "Espèce": "chien",
  "Âge": 4,
  "Poids": 13,
  "Race": "Berger Allemand",
  "Température": 39.2,
  "Respiration": 26.2,
  "Pulse": 127.2,
  "Intensité_activité": 0,
  "Score_sommeil": 5
}
