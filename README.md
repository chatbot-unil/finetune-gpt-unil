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

## Utilisation

1. Activer l'environnement virtuel avec `source venv/bin/activate`
2. Lancer le script de preparation des questions reponses avec `python prepare_dataset_students.py`
3. Lancer le script de finetune avec `python finetune_gpt.py`
4. Ensuite il est possible de tester le modèle finetuner avec `python test_fineturned_model.py "question"`

## Préparation des questions/réponses

### 1. Questions avec plusieurs nombres

Pour commencer j'ai décider de construire les questions depuis un modèle comme ceci :

```python
questions_format_filiere = "Combien y a-t-il d'étudiants en {} pour la filière {} ?"
answers_format_filiere = "Il y a {} femmes, {} hommes et {} étudiants au total en {} pour la filière {}."

questions_format = "Combien y a-t-il d'étudiants en {} ?"
answers_format = "Il y a {} femmes, {} hommes et {} étudiants au total en {} à l'UNIL."
```

Chaque questions contient donc 2 paramètres, le premier est l'année, le deuxième est la filière. Le modèle va donc générer des questions comme ceci :

`Combien y a-t-il d'étudiantes en 2014 pour la filière HEC ?`

Tandis que les réponses seront générées comme ceci :

`Il y a x femmes, y hommes et z étudiants au total en 2014 pour la filière HEC.`

Ensuite j'ai fait un script qui va générer toutes les questions/réponses possibles en se basant sur les données de l'annuaire statistiques 2021-2022 de l'UNIL, qui seront mise sous forme de csv de cette manière :

```csv
annee; femmes; hommes; etranger; CH; total
2011; 1519; 1085; 556; 2048; 2604
2012; 1555; 1170; 626; 2099; 2725
2013; 1645; 1209; 692; 2162; 2854
2014; 1699; 1270; 734; 2235; 2969
2015; 1735; 1288; 750; 2273; 3023
2016; 1911; 1309; 792; 2428; 3220
2017; 1993; 1375; 858; 2510; 3368
2018; 2112; 1369; 869; 2612; 3481
2019; 2250; 1438; 905; 2783; 3688
2020; 2477; 1506; 979; 3004; 3983
2021; 2578; 1530; 1039; 3069; 4108
```

Le script va générer un json de questions/réponses pour chaques fillières qui ressemble a ceci :

```json
{
        "messages": [
            {
                "role": "system",
                "content": "You are an UNIL assistant. Be polite and helpful and answers precisely"
            },
            {
                "role": "user",
                "content": "Combien y a-t-il d'étudiants en 2011 pour la filière FBM ?"
            },
            {
                "role": "assistant",
                "content": "Il y a {} femmes, {} hommes et {} étudiants au total en 2011 pour la filière FBM"
            }
        ]
},
```

La dernière étape avant de finetuner le modèle sera de regrouper tous les jsons de questions/réponses en un seul json, qui dois être sous la forme de jsonl qui est demander par openAI et surtout il doit être sous cette forme :

```json
{"messages": [{"role": "system", "content": "You are an UNIL assistant. Be polite and helpful and answers precisely"}, {"role": "user", "content": "Combien y a-t-il d'étudiants en 2011 pour la filière FBM ?"}, {"role": "assistant", "content": "Il y a {} femmes, {} hommes et {} étudiants au total en 2011 pour la filière FBM"}]}
```

#### Résultats des questions avec plusieurs nombres



### 2. Questions avec un seul nombre
