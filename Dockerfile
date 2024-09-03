FROM python:3.12-slim

WORKDIR /app

# Copier les requirements et installer les dépendances
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copier tous les fichiers du projet
COPY . .

# Exposer le port (ajuster si nécessaire)
EXPOSE 5000

# Lancer l'application
CMD ["python", "app.py"]