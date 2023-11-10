# finetune-gpt-unil

## Introduction

Ce répo a pour but de tester de finetune GPT3.5 sur un dataset de questions/réponses concernant l'université de Lausanne et de voir si le modèle peut répondre à des questions, et si oui, avec quelle qualité. Et aussi tester des questions un peut différente de l'entraienement pour voir si le modèle peut s'adapter à des questions un peut différentes. Et aussi les limtatations du modèle, est ce que il peut dire qu'il ne sait pas répondre à une question, ou est ce qu'il va répondre n'importe quoi.

## Préréquis

- Python 3.8
- environnement virtuel python
- pip
- git
- un compte openai
- un fichier .env avec
  - OPENAI_API_KEY
  - SYSTEM_MESSAGE
  - MODEL_ID(optional, se créer automatiquement si pas présent)

## Installation

1. Cloner le répo
2. Créer un environnement virtuel python
   1. faire `python -m venv venv` pour créer l'environnement virtuel
   2. activer l'environnement virtuel avec `source venv/bin/activate`
3. Installer les dépendances avec `pip install -r requirements.txt`
