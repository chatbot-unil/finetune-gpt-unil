import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import json
from datetime import datetime
import argparse
import re

parser = argparse.ArgumentParser(description="Test precision of fine-tuned OpenAI model.")
parser.add_argument('--limit', type=int, default=3, help='Limit of fine-tuned models to retrieve')
parser.add_argument('--log', type=str, default='logs/test_precision_model.log', help='Path to log file')

args = parser.parse_args()

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
system_message = os.getenv("SYSTEM_MESSAGE")

client = OpenAI()

def get_last_fine_tuned_models(limit=3):
    list_models = client.fine_tuning.jobs.list(limit=limit)
    models_info = []
    for model in list_models.data:
        if model.status == 'succeeded':
            model_info = (model.fine_tuned_model, model.hyperparameters.n_epochs)
            models_info.append(model_info)
    return models_info

def completions(message, model_id):
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

def load_jsonl_file(path):
    dict_qa = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            json_line = json.loads(line)
            if 'messages' in json_line and len(json_line['messages']) > 1:
               dict_qa[json_line['messages'][1]['content']] = json_line['messages'][2]['content']
    return dict_qa

def extraire_chiffres(texte):
    return re.findall(r'\b\d+\b', texte)

def evaluer_reponse(response, expected_response, tolerance=0.05):
    chiffres_response = [float(n) for n in extraire_chiffres(response)]
    chiffres_expected = [float(n) for n in extraire_chiffres(expected_response)]
    return all(abs(a - b) <= tolerance * max(a, b) for a, b in zip(chiffres_response, chiffres_expected))

def effectuer_un_test(dict_qa, model_id):
    correct = 0
    counter = 0
    for question, expected_response in dict_qa.items():
        response = completions(question, model_id)
        if evaluer_reponse(response, expected_response):
            correct += 1
        counter += 1
        if counter >= 10:
            break
    return correct / counter * 100

def effectuer_tests_pour_modele(model_id, dict_qa):
    n_tests = 10  # Nombre de tests à effectuer pour chaque modèle
    precision_tests = []

    for _ in range(n_tests):
        precision = effectuer_un_test(dict_qa, model_id)
        precision_tests.append(precision)

    return sum(precision_tests) / n_tests

if __name__ == '__main__':
    dict_qa = load_jsonl_file('data/training_data.jsonl')
    resultats = {}
    
    model_info_list = get_last_fine_tuned_models(limit=args.limit)
    modeles_et_epochs = {model_name: epochs for model_name, epochs in model_info_list}

    print(f"Effectue un test pour chaque modèle ({args.limit} modèles maximum)...")
    print(f"Nombre de questions: {len(dict_qa)}")

    formatted_date_for_filename = datetime.now().strftime("%Y-%m-%d")
    log_file_name = f"{args.log}_{formatted_date_for_filename}.log"

    for model_id, epochs in modeles_et_epochs.items():
        precision_moyenne = effectuer_tests_pour_modele(model_id, dict_qa)
        resultats[model_id] = (epochs, precision_moyenne)

    for model_id, (epochs, precision) in resultats.items():
        print(f"Modèle {model_id} ({epochs} Epochs): Précision moyenne = {precision:.2f}%")

