import os
import json
import time
import random
import warnings
import torch
import vllm
import numpy as np
from collections import defaultdict, Counter
from vllm import LLM, SamplingParams
import re
from scipy.ndimage import label, find_objects

GLOBAL_SEED = 42
mymsg = ''

def myclear():
    global mymsg
    mymsg = ''

def myprint(s):
    global mymsg
    mymsg += s + '\n'

def set_all_seeds(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_functions_from_r1(response) -> list:
    # Pattern for def convert(...) format
    def_pattern = r'```python\s*(def p\([^}]+?\n(?:[^`]|`(?!``))*?)```'
    # Pattern for convert=lambda format  
    lambda_pattern = r'```python\s*(p\s*=\s*lambda[^`]+?)```'
    
    def_functions = re.findall(def_pattern, response, re.DOTALL)
    lambda_functions = re.findall(lambda_pattern, response, re.DOTALL)
    
    # Convert lambda functions to def format for consistency
    converted_lambdas = []
    for lambda_func in lambda_functions:
        # Extract the lambda part after "convert="
        lambda_part = lambda_func.split('=', 1)[1].strip()
        # Convert to def format
        def_version = f"def p(j):\n return {lambda_part[6:].strip()}"  # Remove "lambda"
        converted_lambdas.append(def_version)
    
    all_functions = def_functions + converted_lambdas
    return all_functions if all_functions else []

def execute_function_with_test(function_str, test_cases):
    try:
        # Create a safe execution environment
        exec_globals = {
            'numpy': np, 'np': np, 'math': __import__('math'),
            'collections': __import__('collections'),
            'scipy': __import__('scipy'),
            'itertools': __import__('itertools')
        }
        
        # Execute the function definition
        exec(function_str, exec_globals)
        convert_func = exec_globals['convert']
        
        # Test with provided test cases
        all_passed = True
        for test_case in test_cases:
            try:
                result = convert_func(test_case['input'])
                # Basic validation - check if result is a list of lists
                if not isinstance(result, list) or not all(isinstance(row, list) for row in result):
                    all_passed = False
                    break
            except Exception:
                all_passed = False
                break
        
        return all_passed, convert_func if all_passed else None
    except Exception as e:
        return False, None

def minimize_code(code_str):
    lines = code_str.split('\n')
    
    # Remove empty lines and comments
    lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
    
    # For code golf style, minimize spacing and use single characters
    minimized_lines = []
    for line in lines:
        stripped = line.strip()
        if 'def p(' in stripped:
            # Keep function definition minimal
            minimized_lines.append(stripped)
        else:
            # Remove extra spaces, keep minimal indentation
            if stripped:
                # For code golf, use single space indentation
                minimized_lines.append(' ' + stripped)
    
    return '\n'.join(minimized_lines)

def list2str(lst):
    return str(lst).replace(' ', '')

def load_task_data(data_folder):
    task_data = {}
    if not os.path.exists(data_folder):
        print(f"Data folder {data_folder} does not exist")
        return task_data
    
    for filename in os.listdir(data_folder):
        if filename.endswith('.json') and filename.startswith('task'):
            filepath = os.path.join(data_folder, filename)
            task_id = filename.replace('.json', '')
            try:
                with open(filepath, 'r') as f:
                    task_data[task_id] = json.load(f)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return task_data

def get_task_llm_input(task_data):
    prompt = "Given the following training examples, learn the pattern:\n\n"
    train_examples = task_data.get('train', [])
    for i, example in enumerate(train_examples):
        prompt += f"Training Example {i+1}:\n"
        prompt += f"Input: {list2str(example['input'])}\n"
        prompt += f"Output: {list2str(example['output'])}\n\n"
    test_examples = task_data.get('test', [])
    if test_examples:
        prompt += "Test Examples (generate output for these):\n"
        for i, example in enumerate(test_examples):
            prompt += f"Test Input {i+1}: {list2str(example['input'])}\n"
            if 'output' in example:
                prompt += f"Expected Output {i+1}: {list2str(example['output'])}\n"
    return prompt, test_examples


system_prompt = """You are an expert at solving abstraction reasoning problems using
minimal code.
Your task: Analyze training examples and create an ultra-compact Python
function that transform input matrices to output matrices.

CRITICAL REQUIREMENTS: 
1. Write in CODE GOLF style - use shortest
possible code 
2. Import libraries INSIDE functions only 
3. Use single letters for variables when possible 
4. Minimize whitespace and comments
5. Each solution must use a DIFFERENT approach

Pattern types: rotation, reflection, object detection, color mapping,
filling, symmetry, etc.

Here are some CODE GOLF examples of ultra-minimal functions:

Example 1 - Grid transformation:

    p=lambda j,A=range(9):[[j[r//3][c//3]and j[r%3][c%3]for c in A]for r in A]

Example 2 - Color doubling with conditional:

    p=lambda j:[[c*2 for c in r]for r in j+(j[:3],j[2:5])[j[1]!=j[4]]]

Example 3 - Matrix operations:

     import numpy as np
     a=n.array(i)
     return(a+a.T).tolist()

Example 4 - Pattern filling:

    [[max(i[r][c],i[r-1][c],i[r][c-1])if r*c else i[r][c]for c in range(len(i[0]))]for r in range(len(i))]

Format your response with exactly THREE solutions using an ultra-compact
style. You can use either def format OR lambda format (both are
acceptable):

SOLUTION 1:

    def p(j):
     return # Ultra-minimal approach 1

SOLUTION 2:

    def p(j):
     return # Ultra-minimal lambda approach 2

SOLUTION 3:

    def p(j):
     return # Ultra-minimal approach 3
"""




def main():
    data_folder = "./data_cp"
    output_folder = "./qwen_base_thinking"
    os.makedirs(output_folder, exist_ok=True)
    
    # Set up environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    set_all_seeds()
    warnings.simplefilter('ignore')
    fake_mode = False
    
    if fake_mode:
        print("Running in fake mode - skipping LLM initialization")
        # For testing without actual LLM
        llm = None
        tokenizer = None
    else:
        llm_model_pth = 'Qwen/Qwen3-4B-Thinking-2507'
        MAX_NUM_SEQS = 4
        MAX_MODEL_LEN = 8196 * 3
        
        llm = LLM(
            llm_model_pth,
            dtype="bfloat16",
            max_num_seqs=MAX_NUM_SEQS,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            seed=2024,
        )
        tokenizer = llm.get_tokenizer()
    task_data = load_task_data(data_folder)
    print(f"Loaded {len(task_data)} tasks")
    
    # Process each task
    for task_id, data in task_data.items():
        print(f"\nProcessing {task_id}...")
        
        try:
            task_prompt, test_cases = get_task_llm_input(data)
            full_prompt = system_prompt + "\n\n" + task_prompt
            ## JUST for testing
            if fake_mode:
                golf_responses = [
                    """
SOLUTION 1:
```python
def convert(i):
 return [[c+1 if c<9 else 1 for c in r] for r in i]
```

SOLUTION 2:
```python  
def convert(i):
 return [r[::-1] for r in i[::-1]]
```

SOLUTION 3:
```python
def convert(i):
 import numpy as n
 return n.rot90(i).tolist()
```
""",
                    """
SOLUTION 1:
```python
def convert(i):
 return [[c*2%10 for c in r]for r in i]
```

SOLUTION 2:
```python
def convert(i):
 return [[i[c][r]for c in range(len(i))]for r in range(len(i[0]))]
```

SOLUTION 3:  
```python
def convert(i):
 return [[max(r)if c==0 else c for c in r]for r in i]
```
"""
                ]
                response = random.choice(golf_responses)
            else:
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens= MAX_MODEL_LEN,
                    # max_tokens=2048,
                    seed=42
                )
                
                outputs = llm.generate([full_prompt], sampling_params)
                response = outputs[0].outputs[0].text
                # print(response)
            functions = extract_functions_from_r1(response)
            print(f"Extracted {len(functions)} functions for {task_id}")
            
            # Try each function in random order as we have no priority
            working_code = ""
            if functions:
                random.shuffle(functions)  
                for i, func_code in enumerate(functions):
                    print(f"Testing function {i+1} for {task_id}...")
                    success, _ = execute_function_with_test(func_code, test_cases)
                    
                    if success:
                        print(f"Function {i+1} works! Using this one.")
                        working_code = minimize_code(func_code)
                        break
                    else:
                        print(f"Function {i+1} failed.")
            
            # Save the working code (or empty string if none worked)
            output_file = os.path.join(output_folder, f"{task_id}.py")
            with open(output_file, 'w') as f:
                if working_code:
                    f.write(working_code)
                else:
                    f.write("") 
            
            print(f"Saved result for {task_id} to {output_file}")
            
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            # Save empty file on error
            output_file = os.path.join(output_folder, f"{task_id}.py")
            with open(output_file, 'w') as f:
                f.write("")

if __name__ == "__main__":
    main()