# =========== Dockerfile optimisé pour ton projet ===========
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système (obligatoires pour Ultralytics + OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installer Python deps (Torch CPU + Ultralytics)
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copier ton API Flask
COPY api/ api/

# Copier le modèle
COPY model/ model/

# Copier les templates HTML
COPY api/templates/ api/templates/

# Créer dossier static automatiquement
RUN mkdir -p static

EXPOSE 5000

# Utilisateur non-root (sécurité)
RUN adduser --disabled-password --gecos '' appuser
USER appuser

CMD ["python", "api/app.py"]






# # Image Python légère
# FROM python:3.10-slim

# WORKDIR /app

# # Copier requirements et installer avec timeout plus long
# COPY requirements.txt .
# RUN pip --default-timeout=1000 install --no-cache-dir -r requirements.txt

# # Copier API et modèle
# COPY api/ api/
# COPY model/ model/

# # Exposer port Flask
# EXPOSE 5000

# # Commande pour lancer l'API
# CMD ["python", "api/app.py"]



# # Image Python légère
# FROM python:3.10-slim

# WORKDIR /app

# # Copier requirements et installer
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copier API et modèle
# COPY api/ api/
# COPY model/ model/

# # Exposer port Flask
# EXPOSE 5000

# # Commande pour lancer l'API
# CMD ["python", "api/app.py"]