import os
import json
import time
import random
import warnings
import torch
import vllm
import numpy as np
import pickle
from collections import defaultdict, Counter
from vllm import LLM, SamplingParams
import re
from scipy.ndimage import label, find_objects
from datetime import datetime

GLOBAL_SEED = 42
mymsg = ''

def myclear():
    global mymsg
    mymsg = ''

def myprint(s):
    global mymsg
    mymsg += s + '\n'

def detect_objects(matrix)->dict:
    from collections import defaultdict, Counter
    matrix = np.array(matrix)
    
    # Validate matrix dimensions
    if matrix.shape[0] > 30 or matrix.shape[1] > 30:
        raise ValueError("Matrix dimensions exceed 30x30 maximum")
    
    # Get matrix dimensions
    matrix_height, matrix_width = matrix.shape

    result = {
        'objects': [],
        'counts': defaultdict(int),
        'adjacency': [],
        'overlaps': []
    }
    
    if 0 not in matrix:
        # No background found, treat the entire matrix as a single texture object
        colors = sorted([int(c) for c in np.unique(matrix)])
        color_counts = {str(c): int(np.sum(matrix == c)) for c in colors}
        
        object_info = {
            'id': 0,
            'color': -1,
            'type': 'texture',
            'coordinates': {
                'top_left': (0, 0),
                'bottom_right': (matrix_height-1, matrix_width-1),
                'center': (float(matrix_height/2), float(matrix_width/2)),
                'y_range': (0, matrix_height-1),
                'x_range': (0, matrix_width-1)
            },
            'size': {
                'height': int(matrix_height),
                'width': int(matrix_width),
                'area': int(matrix_height * matrix_width),
                'description': f"{matrix_height}×{matrix_width}"
            },
            'pixels': [[int(y), int(x)] for y in range(matrix_height) for x in range(matrix_width)],
            'bounding_box': {
                'top': 0,
                'left': 0,
                'bottom': int(matrix_height-1),
                'right': int(matrix_width-1),
                'width': int(matrix_width),
                'height': int(matrix_height)
            },
            'colors_present': colors,
            'color_distribution': color_counts
        }
        
        result['objects'].append(object_info)
        result['counts']['texture'] = 1
        return result
    
    # Get unique colors (excluding background color 0)
    colors = sorted([int(c) for c in np.unique(matrix) if c > 0])
    
    # Process each color
    for color in colors:
        mask = (matrix == color)
        labeled_array, num_features = label(mask)
        objects = find_objects(labeled_array)
        
        for i, obj_slice in enumerate(objects):
            obj_mask = labeled_array[obj_slice] == i+1
            obj_pixels = np.argwhere(labeled_array == i+1)
            if len(obj_pixels) == 0:
                continue
                
            min_y, min_x = np.min(obj_pixels, axis=0)
            max_y, max_x = np.max(obj_pixels, axis=0)
            height = max_y - min_y + 1
            width = max_x - min_x + 1
            area = len(obj_pixels)
            
            center_y = (min_y + max_y) / 2
            center_x = (min_x + max_x) / 2
            
            shape_type = determine_shape_type(obj_mask, height, width, area)
            size_str = f"{height}×{width}"
            
            object_info = {
                'id': len(result['objects']),
                'color': int(color),
                'type': shape_type,
                'coordinates': {
                    'top_left': (int(min_y), int(min_x)),
                    'bottom_right': (int(max_y), int(max_x)),
                    'center': (float(center_y), float(center_x)),
                    'y_range': (int(min_y), int(max_y)),
                    'x_range': (int(min_x), int(max_x))
                },
                'size': {
                    'height': int(height),
                    'width': int(width),
                    'area': int(area),
                    'description': size_str
                },
                'pixels': obj_pixels.tolist(),
                'bounding_box': {
                    'top': int(min_y),
                    'left': int(min_x),
                    'bottom': int(max_y),
                    'right': int(max_x),
                    'width': int(width),
                    'height': int(height)
                }
            }
            
            result['objects'].append(object_info)
            result['counts'][shape_type] += 1
    
    return result

def determine_shape_type(obj_mask, height, width, area):
    if area == 1:
        return 'point'
    
    if area == height * width:
        if height == width:
            return 'square'
        else:
            return 'rectangle'
    
    if height == 1 and width > 1:
        return 'horizontal_line'
    
    if width == 1 and height > 1:
        return 'vertical_line'
    return 'irregular'

def display_object_detection_results(results):
    myprint("\n=== OBJECT DETECTION RESULTS ===")
    myprint(f"Found {len(results['objects'])} objects:")
    
    for obj in results['objects']:
        myprint(f"\nObject {obj['id']+1} - {obj['type']} of color {obj['color']}:")
        
        if obj['type'] == 'texture':
            myprint(f"  This is a texture containing colors: {obj['colors_present']}")
            myprint(f"  Color distribution: {obj['color_distribution']}")
        
        myprint(f"  Size: {obj['size']['description']} (area: {obj['size']['area']} pixels)")
        myprint(f"  Bounding Box: top={obj['bounding_box']['top']}, left={obj['bounding_box']['left']}, " +
              f"bottom={obj['bounding_box']['bottom']}, right={obj['bounding_box']['right']}")
        myprint(f"  Center Point: y={obj['coordinates']['center'][0]:.1f}, x={obj['coordinates']['center'][1]:.1f}")
    
    myprint("\n=== OBJECT COUNT BY TYPE ===")
    for shape_type, count in results['counts'].items():
        myprint(f"  {shape_type}: {count}")
    
    myprint("==============================")

def count_colors(matrix)->str:
    """Count the occurrences of each color (0-9) in a matrix"""
    flattened = [cell for row in matrix for cell in row]
    counts = Counter(flattened)
    
    count_str = ", ".join([f"Color {color}: {count}" for color, count in sorted(counts.items())])
    dimensions = f"[{len(matrix)}x{len(matrix[0])}]"
    
    return f"{dimensions} {count_str}"

def set_all_seeds(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_functions_from_r1(response) -> list:
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
        
        # Test with provided test cases and count successes
        passed_count = 0
        total_count = len(test_cases)
        
        for test_case in test_cases:
            try:
                result = convert_func(test_case['input'])
                # Basic validation - check if result is a list of lists
                if isinstance(result, list) and all(isinstance(row, list) for row in result):
                    passed_count += 1
            except Exception:
                continue  # This test case failed
        
        success_rate = passed_count / total_count if total_count > 0 else 0.0
        all_passed = (passed_count == total_count)
        
        return all_passed, convert_func if all_passed else None, success_rate
    except Exception as e:
        return False, None, 0.0

def matrix_similarity(a, b):
    try:
        if not a or not b or len(a) == 0 or len(b) == 0:
            return 0.0
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            return 0.0
        total = len(a) * len(a[0])
        match = 0
        for i in range(len(a)):
            ra, rb = a[i], b[i]
            for j in range(len(ra)):
                if ra[j] == rb[j]:
                    match += 1
        return match / total if total > 0 else 0.0
    except Exception:
        return 0.0

def score_function_on_training(func_str, task_data):
    try:
        trains = task_data.get('train', [])
        if not trains:
            return 0.0
        
        # Create execution environment
        exec_globals = {
            'numpy': np, 'np': np, 'math': __import__('math'),
            'collections': __import__('collections'),
            'scipy': __import__('scipy'),
            'itertools': __import__('itertools')
        }
        
        # Execute the function definition
        exec(func_str, exec_globals)
        convert_func = exec_globals['convert']
        
        sims = []
        for pair in trains:
            try:
                pred = convert_func(pair['input'])
                sims.append(matrix_similarity(pred, pair['output']))
            except Exception:
                sims.append(0.0)
        
        if not sims:
            return 0.0
        return float(sum(sims) / len(sims))
    except Exception:
        return 0.0

def normalize_scores(scores):
    if len(scores) <= 1:
        return [0.0] * len(scores)  # Return zeros if only one score
    
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    
    if std == 0:
        return [0.0] * len(scores)  # All scores are the same
    
    normalized = (scores - mean) / std
    return normalized.tolist()

WARMUP_TRACES = 8  # Number of warmup traces to establish confidence baseline
TOTAL_BUDGET = 16  # Total generation budget
CONFIDENCE_PERCENTILE = 90  # Percentile for confidence threshold
WINDOW_SIZE = 128  # Sliding window size for confidence calculation

def compute_confidence(logprobs):
    confs = []
    for lp in logprobs:
        if lp and len(lp) > 0:
            # Calculate negative average logprob as confidence metric
            avg_logprob = sum([getattr(l, 'logprob', float(l)) for l in lp]) / len(lp)
            confs.append(round(-avg_logprob, 3))
        else:
            confs.append(0.0)
    return confs

def compute_least_grouped(confs, group_size=WINDOW_SIZE):
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]

    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))

    return sliding_means

def weighted_majority_vote(answers, weights):
    if not answers:
        return None

    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)

    if not answer_weights:
        return None

    voted_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
    return voted_answer

def compute_generation_confidence(outputs):
    try:
        if hasattr(outputs, 'outputs') and len(outputs.outputs) > 0:
            out = outputs.outputs[0]
            if hasattr(out, 'logprobs') and out.logprobs:
                logprobs_data = []
                for token_logprob in out.logprobs:
                    if hasattr(token_logprob, 'values'):
                        logprobs_data.append(list(token_logprob.values()))
                    elif isinstance(token_logprob, dict):
                        logprobs_data.append(list(token_logprob.values()))
                    else:
                        logprobs_data.append([token_logprob])
                
                if logprobs_data:
                    confs = compute_confidence(logprobs_data)
                    sliding_confs = compute_least_grouped(confs, WINDOW_SIZE)
                    print("Using sliding")
                    return min(sliding_confs) if sliding_confs else 0.0

            if hasattr(out, 'cumulative_logprob') and out.cumulative_logprob is not None:
                num_tokens = len(out.token_ids) if hasattr(out, 'token_ids') else 1
                # print("Unable to work with new setting")
                # exit()
                return float(out.cumulative_logprob) / max(1, num_tokens)
    except Exception:
        pass
    return float('0.0')

def process_function_trace(func_str, response_text, logprobs_data, task_data, test_cases):
    try:
        # Calculate confidence metrics
        confs = compute_confidence(logprobs_data) if logprobs_data else [0.0]
        sliding_window = compute_least_grouped(confs, WINDOW_SIZE)
        min_conf = min(sliding_window) if sliding_window else 0.0
        
        # Test function validity
        works, _, test_success_rate = execute_function_with_test(func_str, test_cases)
        
        # Score on training data
        train_score = score_function_on_training(func_str, task_data)
        
        trace_data = {
            "function_code": func_str,
            "response_text": response_text,
            "token_count": len(response_text.split()) if response_text else 0,
            "confs": confs,
            "group_confs": sliding_window,
            "min_conf": min_conf,
            "works_on_test": works,
            "test_success_rate": test_success_rate,
            "train_score": train_score,
        }
        
        return trace_data
    except Exception as e:
        return {
            "function_code": func_str,
            "response_text": response_text,
            "token_count": 0,
            "confs": [0.0],
            "group_confs": [0.0],
            "min_conf": 0.0,
            "works_on_test": False,
            "test_success_rate": 0.0,
            "train_score": 0.0,
            "error": str(e)
        }

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
    prompt = "Given the following training examples with object detection analysis, learn the pattern:\n\n"
    
    # Add ALL training examples with object detection
    train_examples = task_data.get('train', [])
    for i, example in enumerate(train_examples):
        prompt += f"Training Example {i+1}:\n"
        prompt += f"Input: {list2str(example['input'])}\n"
        
        # Add object detection for input
        myclear()
        try:
            input_results = detect_objects(example['input'])
            display_object_detection_results(input_results)
            input_detection = mymsg
        except:
            input_detection = "Object detection failed for input."
        
        input_count = count_colors(example['input'])
        prompt += f"Input Stats: {input_count}\n{input_detection}\n"
        
        prompt += f"Output: {list2str(example['output'])}\n"
        
        # Add object detection for output
        myclear()
        try:
            output_results = detect_objects(example['output'])
            display_object_detection_results(output_results)
            output_detection = mymsg
        except:
            output_detection = "Object detection failed for output."
        
        output_count = count_colors(example['output'])
        prompt += f"Output Stats: {output_count}\n{output_detection}\n"
        prompt += "=" * 50 + "\n\n"
    
    # Add test examples (input only) with object detection
    test_examples = task_data.get('test', [])
    if test_examples:
        prompt += "Test Examples (generate output for these):\n"
        for i, example in enumerate(test_examples):
            prompt += f"Test Input {i+1}: {list2str(example['input'])}\n"
            
            # Add object detection for test input
            myclear()
            try:
                test_results = detect_objects(example['input'])
                display_object_detection_results(test_results)
                test_detection = mymsg
            except:
                test_detection = "Object detection failed for test input."
            
            test_count = count_colors(example['input'])
            prompt += f"Test Stats: {test_count}\n{test_detection}\n"
            
            # Include expected output if available for validation
            if 'output' in example:
                prompt += f"Expected Output {i+1}: {list2str(example['output'])}\n"
            prompt += "-" * 30 + "\n"
    
    return prompt, test_examples

# Code golf optimized system prompt with few-shot examples
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
    # Configuration
    data_folder = "./data_cp"
    output_folder = "./qwen_ours"
    os.makedirs(output_folder, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    set_all_seeds()
    warnings.simplefilter('ignore')
    fake_mode = False
    
    if fake_mode:
        print("Running in fake mode - skipping LLM initialization")
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
    
    # Load all task data
    task_data = load_task_data(data_folder)
    print(f"Loaded {len(task_data)} tasks")
    for task_id, data in task_data.items():
        print(f"\nProcessing {task_id} with confidence-based approach...")
        
        try:
            # Create prompt for this task
            task_prompt, test_cases = get_task_llm_input(data)
            full_prompt = system_prompt + "\n\n" + task_prompt
            
            if fake_mode:
                # Enhanced fake mode with confidence simulation
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
                
                # Simulate warmup phase
                warmup_responses = []
                warmup_confs = []
                
                for _ in range(WARMUP_TRACES):
                    response = random.choice(golf_responses)
                    warmup_responses.append(response)
                    # Simulate confidence with realistic range
                    conf = random.uniform(1.5, 3.0)
                    warmup_confs.append(conf)
                
                # Calculate confidence bar
                conf_bar = float(np.percentile(warmup_confs, CONFIDENCE_PERCENTILE))
                print(f"Confidence bar (P{CONFIDENCE_PERCENTILE}): {conf_bar:.4f}")
                
                # Simulate final phase
                final_responses = []
                final_confs = []
                
                remaining_budget = TOTAL_BUDGET - WARMUP_TRACES
                for _ in range(remaining_budget):
                    if random.random() > 0.3:  # 70% chance to continue
                        response = random.choice(golf_responses)
                        final_responses.append(response)
                        conf = random.uniform(0.8, 3.5)
                        final_confs.append(conf)
                    else:
                        break  # Early stopping
                
                print(f"Generated {len(warmup_responses)} warmup + {len(final_responses)} final traces")
                
                # For fake mode, create simple trace objects
                all_traces = []
                for i, resp in enumerate(warmup_responses + final_responses):
                    functions = extract_functions_from_r1(resp)
                    for func_code in functions:
                        trace = {
                            "function_code": func_code,
                            "min_conf": warmup_confs[i] if i < len(warmup_confs) else final_confs[i - len(warmup_confs)],
                            "works_on_test": random.choice([True, False]),
                            "train_score": random.uniform(0.0, 1.0),
                            "test_success_rate": random.uniform(0.0, 1.0)
                        }
                        all_traces.append(trace)
                
                conf_bar = float(np.percentile(warmup_confs, CONFIDENCE_PERCENTILE))
                
                # Select best function for fake mode
                working_funcs = [t for t in all_traces if t['works_on_test']]
                if working_funcs:
                    best_trace = max(working_funcs, key=lambda x: x['train_score'] + x['test_success_rate'])
                    working_code = minimize_code(best_trace['function_code'])
                else:
                    working_code = ""
                
            else:
                # WARMUP PHASE (Confidence Establishment)
                print(f"\n{'-'*40}")
                print("WARMUP PHASE - Establishing Confidence Baseline")
                print(f"{'-'*40}")
                
                warmup_sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=MAX_MODEL_LEN,
                    seed=42,
                    logprobs=10,  
                    n=WARMUP_TRACES
                )
                
                warmup_outputs = llm.generate([full_prompt], warmup_sampling_params)
                warmup_traces = []
                min_confs = []
                
                for i, output in enumerate(warmup_outputs):
                    for choice in output.outputs:
                        response = choice.text
                        logprobs_data = []
                        if choice.logprobs:
                            for token_logprob in choice.logprobs:
                                if hasattr(token_logprob, 'values'):
                                    logprobs_data.append(list(token_logprob.values()))
                                elif isinstance(token_logprob, dict):
                                    logprobs_data.append(list(token_logprob.values()))
                        functions = extract_functions_from_r1(response)
                        for func_code in functions:
                            trace = process_function_trace(func_code, response, logprobs_data, data, test_cases)
                            warmup_traces.append(trace)
                            min_confs.append(trace['min_conf'])
                conf_bar = float(np.percentile(min_confs, CONFIDENCE_PERCENTILE)) if min_confs else 2.0
                
                print(f"Warmup traces: {len(warmup_traces)}")
                print(f"Confidence bar (P{CONFIDENCE_PERCENTILE}): {conf_bar:.4f}")
                # FINAL PHASE (With Early Stopping)
                print(f"\n{'-'*40}")
                print("FINAL PHASE - Generation with Early Stopping")
                print(f"{'-'*40}")
                
                remaining_budget = TOTAL_BUDGET - WARMUP_TRACES
                final_sampling_params = SamplingParams(
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=MAX_MODEL_LEN,
                    seed=43,
                    logprobs=10,
                    n=remaining_budget
                )
                
                final_outputs = llm.generate([full_prompt], final_sampling_params)
                final_traces = []
                
                for i, output in enumerate(final_outputs):
                    for choice in output.outputs:
                        response = choice.text
                        logprobs_data = []
                        if choice.logprobs:
                            for token_logprob in choice.logprobs:
                                if hasattr(token_logprob, 'values'):
                                    logprobs_data.append(list(token_logprob.values()))
                                elif isinstance(token_logprob, dict):
                                    logprobs_data.append(list(token_logprob.values()))
                        functions = extract_functions_from_r1(response)
                        for func_code in functions:
                            trace = process_function_trace(func_code, response, logprobs_data, data, test_cases)
                            final_traces.append(trace)
                
                print(f"Final traces: {len(final_traces)}")
                all_traces = warmup_traces + final_traces
                # CONFIDENCE-BASED VOTING
                print(f"\n{'-'*40}")
                print("CONFIDENCE-BASED FUNCTION SELECTION")
                print(f"{'-'*40}")
                high_conf_traces = [t for t in all_traces if t['min_conf'] >= conf_bar]
                
                print(f"Traces above confidence threshold: {len(high_conf_traces)}/{len(all_traces)}")
                
                if high_conf_traces:
                    voting_functions = []
                    voting_weights = []
                    
                    for trace in high_conf_traces:
                        if trace['works_on_test']:  # Only consider functions that work
                            voting_functions.append(trace['function_code'])
                            # Use combination of confidence and performance as weight
                            weight = trace['min_conf'] * (1 + trace['train_score'] + trace['test_success_rate'])
                            voting_weights.append(weight)
                    
                    if voting_functions:
                        # Perform weighted voting to select best function
                        voted_function = weighted_majority_vote(voting_functions, voting_weights)
                        print(f"Selected function through weighted voting from {len(voting_functions)} candidates")
                        working_code = minimize_code(voted_function)
                    else:
                        working_funcs = [t for t in all_traces if t['works_on_test']]
                        if working_funcs:
                            best_trace = max(working_funcs, key=lambda x: x['train_score'] + x['test_success_rate'])
                            working_code = minimize_code(best_trace['function_code'])
                            print(f"Using fallback function with combined score: {best_trace['train_score'] + best_trace['test_success_rate']:.3f}")
                        else:
                            working_code = ""
                else:
                    print("No traces met confidence threshold - using best available function")
                    if all_traces:
                        working_funcs = [t for t in all_traces if t['works_on_test']]
                        if working_funcs:
                            best_trace = max(working_funcs, key=lambda x: x['train_score'] + x['test_success_rate'])
                            working_code = minimize_code(best_trace['function_code'])
                        else:
                            working_code = ""
                    else:
                        working_code = ""
                
                # Show some confidence statistics
                if all_traces:
                    conf_values = [t['min_conf'] for t in all_traces]
                    # print(f"  Confidence range: {min(conf_values):.4f} - {max(conf_values):.4f}")
                    # print(f"  Mean confidence: {np.mean(conf_values):.4f}")
                
            # Save the working code
            output_file = os.path.join(output_folder, f"{task_id}.py")
            with open(output_file, 'w') as f:
                if working_code:
                    f.write(working_code)
                else:
                    f.write("")  # Empty string if no function worked
            
            print(f"Saved result for {task_id} to {output_file}")
            
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            # Save empty file on error
            output_file = os.path.join(output_folder, f"{task_id}.py")
            with open(output_file, 'w') as f:
                f.write("")

if __name__ == "__main__":
    main()