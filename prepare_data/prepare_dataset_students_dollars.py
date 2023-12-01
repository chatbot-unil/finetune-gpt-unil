import pandas as pd
import json
import os
import sys
import random
import argparse
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning OpenAI model.")
parser.add_argument('--path', type=str, default='../data', help='Path to CSV file or directory containing CSV files')
parser.add_argument('--path_csv', type=str, default='../data/csv', help='Path to CSV file or directory containing CSV files')
parser.add_argument('--path_validating_data', type=str, default='../data/validating_data.jsonl', help='Path to validating data file')
parser.add_argument('--path_training_data', type=str, default='../data/training_data.jsonl', help='Path to training data file')
parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat each question-answer pair')
args = parser.parse_args()

questions_format_filiere = "Combien y a-t-il d'étudiants en {} pour la filière {} ?"
answers_format_filiere = "Il y a {} femmes, {} hommes et {} étudiants au total en {} pour la filière {}."

questions_format = "Combien y a-t-il d'étudiants en {} ?"
answers_format = "Il y a {} femmes, {} hommes et {} étudiants au total en {} à l'UNIL."

questions_format_filiere = [
    "Combien y a-t-il d'étudiants en {} pour la filière {} ?",
]

answers_format_filiere = [
    "Il y a {} femmes, {} hommes et {} étudiants au total en {} pour la filière {}.",
]

questions_format = [
    "Combien y a-t-il d'étudiants en {} ?",
]

answers_format = [
    "Il y a {} femmes, {} hommes et {} étudiants au total en {} à l'UNIL.",
]

questions_format_filiere_separated = [
    "Combien y a-t-il d'étudiantes en {} pour la filière {} ?",
    "Combien y a-t-il d'étudiants en {} pour la filière {} ?",
    "Combien y a-t-il d'étudiants au total en {} pour la filière {} ?",
]

answers_format_filiere_separated = [
    "Il y a ${} étudiantes pour la filière {} en {}.",
    "Il y a ${} étudiants pour la filière {} en {}.",
    "Il y a ${} étudiants au total pour la filière {} en {}.",
]

def load_data(path):
    """Charge les données depuis un fichier CSV et retourne un DataFrame"""
    try:
        data = pd.read_csv(path, delimiter=';', skipinitialspace=True).dropna()
        return data
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

def create_sentences_from_data(data, filiere):
    sentences = []

    for _, row in data.iterrows():
        femmes = int(row['femmes'])
        hommes = int(row['hommes'])
        total = int(row['total'])
        annee = int(row['annee'])

        if filiere != 'TOTAL':
            for q_template, a_template in zip(questions_format_filiere_separated, answers_format_filiere_separated):
                question = q_template.format(annee, filiere)
                if 'étudiantes' in a_template:
                    answer = a_template.format(femmes, filiere, annee)
                elif 'étudiants' in a_template and 'total' not in a_template:
                    answer = a_template.format(hommes, filiere, annee)
                else:
                    answer = a_template.format(total, filiere, annee)

                sentences.append((question, answer))
        else:
            question = questions_format[0].format(annee)
            answer = answers_format[0].format(femmes, hommes, total, annee)

            sentences.append((question, answer))

    return sentences

def get_filiere(path):
    """Retourne le nom de la filière depuis le chemin du fichier CSV"""
    filename_with_extension = path.split('/')[-1]
    filiere = filename_with_extension.split('.')[0]
    return filiere

def create_training_data(sentences, system_message, repeat_times=1):
    """Creates training data in chat-completion format for each question-answer pair, repeated 'repeat_times' times."""
    training_data = []
    for _ in range(repeat_times):  # Repeat the process 'repeat_times' times
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
    path = f'{args.path}/json/{filiere}.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def merge_json_files(path_training_data, path_validating_data, jsonpath=f'{args.path}/json'):
    json_files = os.listdir(jsonpath)
    training_data = []
    validating_data = []
    for json_file in json_files:
        with open(f'{jsonpath}/{json_file}', 'r', encoding='utf-8') as f:
            data = json.load(f)
            training_data.extend(data)

    validating_data = training_data[:int(len(training_data) * 0.2)]

    random.shuffle(training_data)
    random.shuffle(validating_data)

    with open(path_training_data, 'w', encoding='utf-8') as f:
        for example in training_data:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')

    with open(path_validating_data, 'w', encoding='utf-8') as f:
        for example in validating_data:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')

def process_csv_file(csv_path, system_message, repeat_times=1):
    filiere = get_filiere(csv_path)
    data = load_data(csv_path)
    if data is not None:
        sentences = create_sentences_from_data(data, filiere)
        training_data = create_training_data(sentences, system_message, repeat_times=repeat_times)
        save_json(training_data, filiere)

if __name__ == '__main__':
    repeat_times = args.repeat

    path = args.path_csv
    system_message = "Tu es un data scientist. On te présente des données concernant les étudiants inscrits au semestre d’automne, par faculté selon le sexe. Les valeurs statistiques sont précedées d'un $."

    if os.path.isfile(path) and path.endswith(".csv"):
        # If the path is a CSV file, process it
        process_csv_file(path, system_message)
        merge_json_files(args.path_training_data, args.path_validating_data)
    elif os.path.isdir(path):
        # If the path is a directory, process all CSV files in it
        found_csv = False
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                found_csv = True
                csv_path = os.path.join(path, filename)
                process_csv_file(csv_path, system_message, repeat_times=repeat_times)
                merge_json_files(args.path_training_data, args.path_validating_data)
        if not found_csv:
            print("No CSV files found in the directory.")
            sys.exit(1)
    else:
        print("The provided path is neither a CSV file nor a directory.")
        sys.exit(1)
    
