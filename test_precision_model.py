import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import json
import argparse
import re

parser = argparse.ArgumentParser(description="Test precision of fine-tuned OpenAI model.")
parser.add_argument('--limit', type=int, default=3, help='Limit of fine-tuned models to retrieve')
parser.add_argument('--log', type=str, default='logs/precision_tests.jsonl', help='Log file name')
parser.add_argument('--nb_tests', type=int, default=10, help='Number of questions to test')
parser.add_argument('--tolerance', type=float, default=0.0, help='Tolerance for precision evaluation')

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

def evaluer_reponse(response, expected_response):
    chiffres_response = [float(n) for n in extraire_chiffres(response)]
    chiffres_expected = [float(n) for n in extraire_chiffres(expected_response)]
    return all(abs(a - b) <= args.tolerance * max(a, b) for a, b in zip(chiffres_response, chiffres_expected))


def effectuer_un_test(dict_qa, model_id):
    test_results = []
    for question, expected_response in dict_qa.items():
        response = completions(question, model_id)
        chiffres_response = extraire_chiffres(response)
        chiffres_expected = extraire_chiffres(expected_response)
        precision = evaluer_reponse(response, expected_response)
        test_results.append({
            "question": question,
            "response": response,
            "expected_response": expected_response,
            "chiffres_response": chiffres_response,
            "chiffres_expected": chiffres_expected,
            "precision": precision
        })
        if len(test_results) >= 10:
            break
    return test_results

def effectuer_tests_pour_modele(model_id, dict_qa, epochs):
    test_results = effectuer_un_test(dict_qa, model_id)
    precision_moyenne = (sum(test["precision"] for test in test_results) / len(test_results)) * 100
    return {
        "model_id": model_id,
        "epochs": epochs,
        "precision_moyenne": precision_moyenne,
        "tests": test_results
    }

def write_logs_to_file(log_data, log_file_name):
    with open(log_file_name, 'a', encoding='utf-8') as file:
        json.dump(log_data, file, indent=4, ensure_ascii=False)
        file.write("\n")

if __name__ == '__main__':
    dict_qa = load_jsonl_file('data/training_data.jsonl')
    model_info_list = get_last_fine_tuned_models(limit=args.limit)
    modeles_et_epochs = {model_name: epochs for model_name, epochs in model_info_list}
    resultats = {}

    for model_id, epochs in modeles_et_epochs.items():
        log_data = effectuer_tests_pour_modele(model_id, dict_qa, epochs)
        resultats[model_id] = (epochs, log_data["precision_moyenne"])
        write_logs_to_file(log_data, args.log)

    # Affichage des résultats
    for model_id, (epochs, precision_moyenne) in resultats.items():
        print(f"Modèle {model_id} ({epochs} Epochs): Précision moyenne = {precision_moyenne:.2f}%")

