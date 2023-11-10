import pandas as pd
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

questions_format_filiere = "Combien y a-t-il d'étudiants en {} pour la filière {} ?"
answers_format_filiere = "Il y a {} femmes, {} hommes et {} étudiants au total en {} pour la filière {}."

questions_format = "Combien y a-t-il d'étudiants en {} ?"
answers_format = "Il y a {} femmes, {} hommes et {} étudiants au total en {} à l'UNIL."

def load_data(path):
    """Charge les données depuis un fichier CSV et retourne un DataFrame"""
    try:
        data = pd.read_csv(path, delimiter=';', skipinitialspace=True).dropna()
        return data
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

def create_sentences_from_data(data, filiere):
    """Créé un tuple (question, réponse) pour chaque ligne du DataFrame en utilisant le format spécifié"""
    sentences = []
    for _, row in data.iterrows():
        if filiere != 'TOTAL':
            question = questions_format_filiere.format(row['annee'], filiere)
            answer = answers_format_filiere.format(row['femmes'], row['hommes'], row['total'], row['annee'], filiere)
        else:
            question = questions_format.format(row['annee'])
            answer = answers_format.format(row['femmes'], row['hommes'], row['total'], row['annee'])
        sentences.append((question, answer))
    return sentences

def get_filiere(path):
    """Retourne le nom de la filière depuis le chemin du fichier CSV"""
    filename_with_extension = path.split('/')[-1]
    filiere = filename_with_extension.split('.')[0]
    return filiere

def create_training_data(sentences, system_message):
    """Creates training data in chat-completion format for each question-answer pair."""
    training_data = []
    for question, answer in sentences:
        entry = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        training_data.append(entry)
    return training_data

def save_json(data, filiere):
    """Sauvegarde les données dans un fichier JSON"""
    path = f'data/json/{filiere}.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def process_csv_file(csv_path, system_message):
    filiere = get_filiere(csv_path)
    data = load_data(csv_path)
    if data is not None:
        sentences = create_sentences_from_data(data, filiere)
        training_data = create_training_data(sentences, system_message)
        save_json(training_data, filiere)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_csv_or_directory>")
        sys.exit(1)

    path = sys.argv[1]
    system_message = os.getenv("SYSTEM_MESSAGE")

    if os.path.isfile(path) and path.endswith(".csv"):
        # If the path is a CSV file, process it
        process_csv_file(path, system_message)
    elif os.path.isdir(path):
        # If the path is a directory, process all CSV files in it
        found_csv = False
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                found_csv = True
                csv_path = os.path.join(path, filename)
                process_csv_file(csv_path, system_message)
        if not found_csv:
            print("No CSV files found in the directory.")
            sys.exit(1)
    else:
        print("The provided path is neither a CSV file nor a directory.")
        sys.exit(1)
    
