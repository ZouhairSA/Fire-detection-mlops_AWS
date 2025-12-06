# 🔥 FireVision – Détection Automatique d’Incendie avec Intelligence Artificielle

FireVision est un système avancé de **détection d’incendie** utilisant un modèle
de vision par ordinateur basé sur **YOLOv8**.

Le projet permet de :
- analyser des images
- détecter la présence de flammes
- retourner un niveau de confiance
- exporter un rapport PDF
- utiliser une interface Web moderne (Dark Mode)

---

## 🚀 Technologies utilisées

| Domaine | Technologie |
|---------|-------------|
| IA / Vision | YOLOv8, Ultralytics |
| Backend | Flask / Python |
| Frontend | HTML5, CSS3, Bootstrap |
| Rapport | génération PDF |
| Déploiement | local / serveur |

---

## 🔧 Fonctionnement

1️⃣ L’utilisateur importe une image  
2️⃣ L’image est envoyée au serveur Flask  
3️⃣ Le modèle YOLOv8 analyse l’image  
4️⃣ Le résultat renvoie :
- présence de feu
- niveau de confiance
- position du feu (si besoin)

5️⃣ L’utilisateur peut :
- afficher le résultat
- télécharger un rapport PDF
- consulter un rapport texte

---

## 🎯 Objectifs du projet

- aider à la détection précoce d’incendies
- réduire les risques
- fournir un outil simple et rapide
- démontrer l’utilisation des modèles YOLO dans un cas réel

---

## 🧠 Améliorations possibles

✅ Détection vidéo en temps réel  
✅ Alertes SMS / Email  
✅ Déploiement sur Raspberry Pi  
✅ Ajout d’autres classes (fumée, chaleur, etc.)  

---

## 📦 Installation

```bash
pip install ultralytics flask reportlab
