"""
    FEATURE: ChartQA Evaluation of ChartMoE, with/without PoT(Program of Thoughts)
    AUTHOR: Brian Qu
    URL: https://arxiv.org/abs/2409.03277
"""
from chartmoe import ChartMoE_Robot
from chartmoe.utils.custom_path import ChartQA_ROOT, ChartQA_TEST_IMG_ROOT

import os, sys, json, re, io
import argparse
import torch
from tqdm import tqdm
from typing import Optional
from prettytable import PrettyTable

def relaxed_acc(prediction: str, target: str,
                    max_relative_change: float = 0.05) -> bool:

    def _to_float(text: str) -> Optional[float]:
        try:
            match = re.search(r'[\d.]+', text.replace(',', ''))
            if match: return float(match.group())
            return None
        except ValueError:
            return None
        
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    
    if prediction_float is not None and target_float is not None:
        if target_float == 0:
            relative_change = abs(prediction_float - target_float)
        else:
            relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        lp = prediction.lower() 
        tp = target.lower()

        if ("yes" in lp and "yes" in tp) or ("no" in lp and "no" in tp): return True
        if lp in tp: return True
        return lp == tp

def evaluate_relaxed_accuracy(entries, margin=0.05):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_acc(elem['answer'].strip(), ann, margin)
            for ann in elem['annotation']
        ])
        scores.append(score)

    return sum(scores) / len(scores)

def execute_python_code(code):
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    status = True
    try:
        exec(code)
    except Exception as e:
        status = False
    finally:
        sys.stdout = old_stdout

    if status:
        output = new_stdout.getvalue()
    else:
        output = None
    return output, status

def extract_python_content(text):
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

class ChartQATester:

    def __init__(self, pot=False, pot_idx=0):
        # ChartQA root
        self.root = ChartQA_ROOT
        self.vis_root = ChartQA_TEST_IMG_ROOT

        self.robot = ChartMoE_Robot()
        self.prompt = '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase. {}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
        pot_prompts = [
            '[UNUSED_TOKEN_146]user\nPlease give the program of thought. {}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n',
            '[UNUSED_TOKEN_146]user\nPlease give the program of thought in python code. Use print function to output the answer in the end. {}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n',
        ]
        self.pot_prompt = pot_prompts[pot_idx]
        
        self.pot = pot
        if self.pot:
            self.robot.reset_prompt(prompt=self.pot_prompt)
        else:
            self.robot.reset_prompt(prompt=self.prompt)
    
    def reset_prompt(self, p):
        self.system_prompt = p
    
    def infer_all_answers(self, output_path):

        os.makedirs(output_path, exist_ok=True)
        print(f"Result will be saved at: {output_path}")
        
        part_acc = []
        for part_name in ['human', 'augmented']:
            part_json = os.path.join(output_path, f"{part_name}.json")
            if os.path.exists(part_json):
                print(f"Load result from: {part_json}")
                part = json.load(open(part_json, 'r'))
            else:
                part = []
                samples = json.load(open(self.root+f'test/test_{part_name}.json')) 
                for q in tqdm(samples):
                    im_path = os.path.join(self.vis_root, q['imgname'])
                    question = q['query']

                    with torch.cuda.amp.autocast():
                        response, _ = self.robot.chat(
                            image_path=im_path, 
                            question=question,
                            max_new_tokens=500,
                            num_beams=1,
                        )
                    if self.pot:
                        extraced_result = extract_python_content(response)
                        if extraced_result:
                            code = extraced_result[0]
                        else:
                            code = response
                        response, status = execute_python_code(code)
                        
                        if not status:
                            response = "error running..."
                        response = response.replace("True","Yes").replace("False","No")
                        response = response.strip()
                        part.append({
                            'image': im_path,
                            'query': question,
                            'answer': response,
                            'annotation': q['label'],
                            'code': code
                        }) 
                    else:
                        part.append({
                            'image': im_path,
                            'query': question,
                            'answer': response,
                            'annotation': q['label'] 
                        }) 
                with open(part_json, 'w') as f:
                    json.dump(part, f, indent=4)
            part_acc.append(part)
        
        table = PrettyTable()
        table.field_names = ["@AP", "0.05", "0.1", "0.2"]  
        human_row = ["Human"]
        augmented_row = ["Augmented"]
        averaged_row = ["Averaged"]
        for ap in [0.05, 0.1, 0.2]:
            part_acc_ap = [evaluate_relaxed_accuracy(p, ap) for p in part_acc]
            human_acc = part_acc_ap[0]
            augmented_acc = part_acc_ap[1]
            averaged_acc = (human_acc + augmented_acc) / 2
            human_row.append(human_acc)
            augmented_row.append(augmented_acc)
            averaged_row.append(averaged_acc)
     
        table.add_row(human_row)
        table.add_row(augmented_row)
        table.add_row(averaged_row)

        table_path = os.path.join(output_path, 'table.txt')
        with open(table_path, 'w') as f:
            f.write(str(table))

        print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument('--pot', action='store_true')
    parser.add_argument('--pot_idx', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    tester = ChartQATester(pot=args.pot, pot_idx=args.pot_idx)
    tester.infer_all_answers(output_path=args.save_path)