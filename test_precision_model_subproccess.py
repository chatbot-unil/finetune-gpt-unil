import subprocess
import argparse
import os
import json
from datetime import datetime

parser = argparse.ArgumentParser(description="Test precision of fine-tuned OpenAI model process.")
parser.add_argument('--limit', type=int, default=2, help='Limit of fine-tuned models to retrieve')
parser.add_argument('--times', type=int, default=10, help='Number of times to run the process')
parser.add_argument('--nb_tests', type=int, default=20, help='Number of questions to test')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for completion')
parser.add_argument('--script', type=str, default='test_precision_model.py', help='Script to run')
parser.add_argument('--results', type=str, default='logs/', help='Log file name')
parser.add_argument('--purpose', type=str, default='test', help='Purpose of the test')
parser.add_argument('--log', type=str, default='logs/', help='Log file name')

args = parser.parse_args()

def run_process_x_times(command, x):
    for _ in range(x):
        subprocess.run(command, shell=True)

def make_moyenne(json_file):
    results = {}
    with open(f'{json_file}', 'r', encoding='utf-8') as f:
        results = json.load(f)
    for result in results:
        model_id = result['model_id']
        epochs = result['epochs']
        precisions = result['precisions']
        moyenne = sum(precisions) / len(precisions) if precisions else 0
        result['precision_moyenne'] = moyenne
        precisions = []
        print(f"Model {model_id} with {epochs} epochs, precision moyenne sur {args.times} x {args.nb_tests} tests: {moyenne}")
    
    results.append({
        'purpose': args.purpose,
        'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    })
    
    with open(f'{json_file}', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

command = f"python3 {args.script} --limit {args.limit} --nb_tests {args.nb_tests} --temperature {args.temperature} --results {args.results} --purpose \"{args.purpose}\" --log \"{args.log}\""

run_process_x_times(command, args.times)
make_moyenne(args.results)