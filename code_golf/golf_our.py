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

def detect_objects(matrix)->dict:
    from collections import defaultdict, Counter
    
    # Convert input to numpy array if it isn't already
    matrix = np.array(matrix)
    
    # Validate matrix dimensions
    if matrix.shape[0] > 30 or matrix.shape[1] > 30:
        raise ValueError("Matrix dimensions exceed 30x30 maximum")
    
    # Get matrix dimensions
    matrix_height, matrix_width = matrix.shape
    
    # Initialize result structure
    result = {
        'objects': [],
        'counts': defaultdict(int),
        'adjacency': [],
        'overlaps': []
    }
    
    # Check if the matrix contains any background (0 values)
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
    # Pattern for def convert(...) format
    def_pattern = r'```python\s*(def p\([^}]+?\n(?:[^`]|`(?!``))*?)```'
    # Pattern for convert=lambda format  
    lambda_pattern = r'```python\s*(p\s*=\s*lambda[^`]+?)```'
    
    def_functions = re.findall(def_pattern, response, re.DOTALL)
    lambda_functions = re.findall(lambda_pattern, response, re.DOTALL)
    
    # Convert lambda functions to def format for consistency
    converted_lambdas = []
    for lambda_func in lambda_functions:
        lambda_part = lambda_func.split('=', 1)[1].strip()
        # Convert to def format
        def_version = f"def p(j):\n return {lambda_part[6:].strip()}" 
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
    """Score a function candidate by how well it maps training inputs to outputs"""
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
        return [0.0] * len(scores) 
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    if std == 0:
        return [0.0] * len(scores)  # All scores are the same
    
    normalized = (scores - mean) / std
    return normalized.tolist()

def compute_generation_confidence(outputs):
    try:
        if hasattr(outputs, 'outputs') and len(outputs.outputs) > 0:
            out = outputs.outputs[0]
            if hasattr(out, 'logprobs') and out.logprobs:
                chosen_lps = []
                for pos, top_dict in enumerate(out.logprobs):
                    if isinstance(top_dict, dict) and len(top_dict) > 0:
                        try:
                            best_lp = max(tp.logprob if hasattr(tp, 'logprob') else float(tp) for tp in top_dict.values())
                        except Exception:
                            best_lp = max(float(v) for v in top_dict.values())
                        chosen_lps.append(best_lp)
                if chosen_lps:
                    return float(sum(chosen_lps) / len(chosen_lps))
            
            # Fallback: if cumulative logprob exists
            if hasattr(out, 'cumulative_logprob') and out.cumulative_logprob is not None:
                print("Dont exist")
                num_tokens = len(out.token_ids) if hasattr(out, 'token_ids') else 1
                return float(out.cumulative_logprob) / max(1, num_tokens)
    except Exception:
        pass
    return float('nan')

def minimize_code(code_str):
    lines = code_str.split('\n')
    
    # Remove empty lines and comments
    lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
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
    
    # Create output folder if it doesn't exist
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
    
    # Load all task data
    task_data = load_task_data(data_folder)
    print(f"Loaded {len(task_data)} tasks")
    
    # Process each task
    for task_id, data in task_data.items():
        print(f"\nProcessing {task_id}...")
        
        try:
            task_prompt, test_cases = get_task_llm_input(data)
            full_prompt = system_prompt + "\n\n" + task_prompt
            print("Input prompt:",full_prompt)
            
            if fake_mode:
                # Generate more realistic dummy responses for testing with code golf style
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
                    max_tokens=MAX_MODEL_LEN,
                    seed=42,
                    logprobs=5 
                )
                
                outputs = llm.generate([full_prompt], sampling_params)
                response = outputs[0].outputs[0].text
                # print(response)
                generation_confidence = compute_generation_confidence(outputs[0])
            
            functions = extract_functions_from_r1(response)
            print(f"Extracted {len(functions)} functions for {task_id}")
            # print(f"Generation confidence: {generation_confidence if not np.isnan(generation_confidence) else 'N/A'}")
            function_scores = []  
            
            if functions:
                for i, func_code in enumerate(functions):
                    print(f"Testing function {i+1} for {task_id}...")
                    works, _, test_success_rate = execute_function_with_test(func_code, test_cases)
                    train_score = score_function_on_training(func_code, data)
                    
                    function_scores.append((func_code, train_score, works, test_success_rate))
                    print(f"Function {i+1}: works={works}, train_score={train_score:.3f}, test_success_rate={test_success_rate:.3f}")
            
            if function_scores:
                # Extract individual metric arrays
                train_scores = [f[1] for f in function_scores]
                test_success_rates = [f[3] for f in function_scores]
                
                # if not np.isnan(generation_confidence):
                #     confidence_scores = [generation_confidence] * len(function_scores)  # Same for all from this generation
                # else:
                confidence_scores = [0.0] * len(function_scores)
                
                # Apply standard normalization to each metric
                norm_train_scores = normalize_scores(train_scores)
                norm_test_success = normalize_scores(test_success_rates)  
                norm_confidence = normalize_scores(confidence_scores)
                
                # Calculate combined scores (sum of normalized metrics)
                combined_scores = []
                for i in range(len(function_scores)):
                    combined_score = norm_train_scores[i] + norm_test_success[i] + norm_confidence[i]
                    combined_scores.append(combined_score)
                    print(f"Function {i+1} combined score: {combined_score:.3f} " +
                          f"(train: {norm_train_scores[i]:.2f}, test: {norm_test_success[i]:.2f}, conf: {norm_confidence[i]:.2f})")
                
                # Find function with highest combined score
                best_idx = np.argmax(combined_scores)
                best_func_code = function_scores[best_idx][0]
                best_combined_score = combined_scores[best_idx]
                best_train_score = train_scores[best_idx]
                best_works = function_scores[best_idx][2]
                
                # print(f"Selected function {best_idx+1} with combined score: {best_combined_score:.3f}")
                # print(f"  Training score: {best_train_score:.3f}, Works on tests: {best_works}")
            else:
                best_func_code = ""
                print("No valid functions extracted")
            
            working_code = minimize_code(best_func_code) if best_func_code else ""
            
            # Save the working code (or empty string if none worked)
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