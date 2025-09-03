## Soving ARC2 Problems using Deepseek R1

import sys
GLOBAL_SEED = 42
mymsg = ''#用于存储消息

def myclear():
    """
    Clear the global message string used for accumulating output messages.
    """
    global mymsg
    mymsg = ''


def myprint(s):
    """
    Append a string to the global message accumulator, followed by a newline.
    Args:
        s (str): The string to append.
    """
    global mymsg
    mymsg += s + '\n'
    
def detect_objects(matrix) -> dict:
    """
    Detect objects in a 2D matrix, including rectangles, points, and irregular shapes.
    If the matrix doesn't contain any background (0 values), treat the entire matrix as a single "texture" object.

    Args:
        matrix (List[List[int]]): A 2D integer matrix with values 0-9, where 0 typically
            represents background and 1-9 represent different objects or colors. Matrix should be at most 30x30.

    Returns:
        dict: A dictionary containing detected objects with their properties and relationships.
    """
    import numpy as np
    from collections import defaultdict, Counter
    from scipy.ndimage import label, find_objects
    
    # Convert input to numpy array if it isn't already
    matrix = np.array(matrix)
    
    # Validate matrix dimensions
    # 限制矩阵大小不超过30x30
    if matrix.shape[0] > 30 or matrix.shape[1] > 30:
        raise ValueError("Matrix dimensions exceed 30x30 maximum")
    
    # Get matrix dimensions
    matrix_height, matrix_width = matrix.shape
    
    # Initialize result structure
    result = {
        'objects': [],#对象信息，包括颜色、位置等
        'counts': defaultdict(int),#每种形状有多少个
        'adjacency': [],  # 对象之间的邻接关系
        'overlaps': []  # 对象之间的重叠关系
    }
    
    # Check if the matrix contains any background (0 values)
    if 0 not in matrix:#如果这个形状不是黑色的，就进行统计
        # No background found, treat the entire matrix as a single texture object
        # 出现过的颜色 排序
        colors = sorted([int(c) for c in np.unique(matrix)])
        # 统计每种颜色出现的次数
        color_counts = {str(c): int(np.sum(matrix == c)) for c in colors}
        
        # Create object info for the entire texture
        object_info = {
            'id': 0,
            'color': -1,  # Special value indicating multiple colors
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
                'area': int(matrix_height * matrix_width),#面积
                'description': f"{matrix_height}×{matrix_width}"
            },
            # 所有像素点的坐标
            'pixels': [[int(y), int(x)] for y in range(matrix_height) for x in range(matrix_width)],
            'bounding_box': {
                'top': 0,
                'left': 0,
                'bottom': int(matrix_height-1),
                'right': int(matrix_width-1),
                'width': int(matrix_width),
                'height': int(matrix_height)
            },
            'colors_present': colors,  # List of colors in the texture
            'color_distribution': color_counts  # Count of each color
        }
        
        result['objects'].append(object_info)
        result['counts']['texture'] = 1
        
        return result
    
    # Get unique colors (excluding background color 0)
    colors = sorted([int(c) for c in np.unique(matrix) if c > 0])
    
    # Process each color
    # 遍历每种颜色
    for color in colors:
        # Create binary mask for current color
        # 为当前颜色创建掩码
        mask = (matrix == color)
        
        # Label connected components
        # 对mask部分的像素，标记连通区域
        labeled_array, num_features = label(mask)
        
        # Find objects for each labeled region
        # 根据标记后的连通数组，找到每个标记区域的边界框
        objects = find_objects(labeled_array)
        
        # 遍历每个连通区域，其中i是连通区域的编号，obj_slice是连通区域的边界框
        for i, obj_slice in enumerate(objects):
            # Extract object properties
            # 用掩码提取边界框内的区域
            obj_mask = labeled_array[obj_slice] == i+1
            # 获取边界框内所有像素的坐标
            obj_pixels = np.argwhere(labeled_array == i+1)
            if len(obj_pixels) == 0:
                continue
                
            # Calculate object dimensions
            # 获取所有像素的各个边界
            min_y, min_x = np.min(obj_pixels, axis=0)
            max_y, max_x = np.max(obj_pixels, axis=0)
            height = max_y - min_y + 1
            width = max_x - min_x + 1
            area = len(obj_pixels)
            
            # Calculate center coordinates
            center_y = (min_y + max_y) / 2
            center_x = (min_x + max_x) / 2
            
            # Determine shape type
            shape_type = determine_shape_type(obj_mask, height, width, area)
            
            # Format size as "height×width"
            size_str = f"{height}×{width}"
            
            # Collect object information with exact coordinates
            # 得到这个连通区域的详细信息
            object_info = {
                'id': len(result['objects']),  # Assign unique ID
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
    
    # Find adjacent objects
    result['adjacency'] = find_adjacent_objects(result['objects'], matrix.shape)
    
    # Find overlapping objects (based on bounding boxes)
    result['overlaps'] = find_overlapping_objects(result['objects'])
    
    return result

# 找到重叠的对象
def find_overlapping_objects(objects) -> list:
    """
    Find pairs of objects whose bounding boxes overlap.
    Args:
        objects (list): List of detected object dictionaries.
    Returns:
        list: List of dictionaries describing overlapping object pairs.
    """
    overlaps = []
    
    # 遍历每一对
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects[i+1:], i+1):
            # 如果两个对象颜色相同，则跳过
            if obj1['color'] == obj2['color']:
                continue
                
            # Get bounding boxes
            bb1 = obj1['bounding_box']
            bb2 = obj2['bounding_box']
            
            # Check for overlap
            # 检查两个区域的上界、下界、左界、右界是否重叠
            if (bb1['left'] <= bb2['right'] and bb1['right'] >= bb2['left'] and
                bb1['top'] <= bb2['bottom'] and bb1['bottom'] >= bb2['top']):
                
                # Calculate overlap area
                # 获得重叠区域的宽度和高度
                overlap_width = min(bb1['right'], bb2['right']) - max(bb1['left'], bb2['left'])
                overlap_height = min(bb1['bottom'], bb2['bottom']) - max(bb1['top'], bb2['top'])
                overlap_area = max(0, overlap_width) * max(0, overlap_height)
                
                # Calculate overlap percentage relative to smaller object
                smaller_area = min(bb1['width'] * bb1['height'], bb2['width'] * bb2['height'])
                overlap_percentage = (overlap_area / smaller_area) if smaller_area > 0 else 0
                
                overlaps.append({
                    'object1': obj1['id'],
                    'object2': obj2['id'],
                    'color1': obj1['color'],
                    'color2': obj2['color'],
                    'type1': obj1['type'],
                    'type2': obj2['type'],
                    'overlap_area': overlap_area,
                    'overlap_percentage': overlap_percentage,
                    'description': get_overlap_description(overlap_percentage)
                })
    
    return overlaps

def determine_shape_type(obj_mask, height, width, area):
    """
    Determine the type of shape based on its properties.
    Args:
        obj_mask (np.ndarray): Binary mask of the object.
        height (int): Height of the bounding box.
        width (int): Width of the bounding box.
        area (int): Area (number of pixels) of the object.
    Returns:
        str: The determined shape type (e.g., 'rectangle', 'circle', etc.).
    """
    # 可选项：点，正方形，长方形，水平线，垂直线，网格，L形，十字，圆形，不规则
    import numpy as np
    
    # Single pixel
    if area == 1:
        return 'point'
    
    # Rectangle check - a rectangle's area equals height × width
    if area == height * width:
        if height == width:
            return 'square'
        else:
            return 'rectangle'
    
    # Line checks with enhanced detection
    if height == 1 and width > 1:
        return 'horizontal_line'
    
    if width == 1 and height > 1:
        return 'vertical_line'
    
    # Grid pattern detection
    if is_grid_pattern(obj_mask):
        return 'grid'
    
    # Check if shape is L-shaped
    if is_l_shape(obj_mask):
        return 'l_shape'
        
    # Check for common patterns
    if is_cross(obj_mask, height, width):
        return 'cross'
    
    # Calculate perimeter
    # 在对象掩码周围添加一圈填充，然后计算周长
    padded = np.pad(obj_mask, 1, mode='constant')
    perimeter_mask = np.logical_and(
        padded[1:-1, 1:-1],
        np.logical_or.reduce([
            ~padded[0:-2, 1:-1],  # up
            ~padded[2:, 1:-1],    # down
            ~padded[1:-1, 0:-2],  # left
            ~padded[1:-1, 2:],    # right
        ])
    )
    perimeter = np.sum(perimeter_mask)
    
    # Circle approximation (using isoperimetric inequality)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    if 0.7 < circularity <= 1.0:
        return 'circle'
    
    # Default to irregular shape
    return 'irregular'

# 判断是否是网格
def is_grid_pattern(mask) -> bool:
    """
    Detect if a pattern resembles a grid with gaps.
    Args:
        mask (np.ndarray): Binary mask of the object.
    Returns:
        bool: True if the pattern is grid-like, False otherwise.
    """
    import numpy as np
    
    h, w = mask.shape
    
    # Too small to be a grid
    if h < 3 or w < 3:
        return False
    
    # Check for alternating pattern in rows and columns
    row_transitions = 0  # 行方向的颜色变化次数
    col_transitions = 0  # 列方向的颜色变化次数
    
    # 遍历每一行，统计行方向的变化次数
    for i in range(h):
        prev_val = mask[i, 0]
        for j in range(1, w):
            if mask[i, j] != prev_val:
                row_transitions += 1
                prev_val = mask[i, j]
    
    # Count vertical transitions
    for j in range(w):
        prev_val = mask[0, j]
        for i in range(1, h):
            if mask[i, j] != prev_val:
                col_transitions += 1
                prev_val = mask[i, j]
    
    # Grid-like pattern should have multiple transitions both horizontally and vertically
    # and the total transitions should be significant relative to the size
    min_transitions = min(h, w) - 1
    # 如果颜色是交替变化的，说明是网格
    return (row_transitions >= min_transitions and 
            col_transitions >= min_transitions and
            (row_transitions + col_transitions) >= (h + w) * 0.4)

# 检查是否为L形
def is_l_shape(mask):
    """
    Check if a mask represents an L shape.
    Args:
        mask (np.ndarray): Binary mask of the object.
    Returns:
        bool: True if the mask is L-shaped, False otherwise.
    """
    import numpy as np
    
    # L-shape has two perpendicular line segments
    # This is a simplified check
    h, w = mask.shape
    
    # Too small to be an L
    if h < 3 or w < 3:
        return False
        
    # Check basic L pattern - one arm along top or bottom, one along left or right
    # 判断每条边被填充的部分是否大于60%
    top_filled = np.sum(mask[0, :]) >= w * 0.6
    bottom_filled = np.sum(mask[h-1, :]) >= w * 0.6
    left_filled = np.sum(mask[:, 0]) >= h * 0.6
    right_filled = np.sum(mask[:, w-1]) >= h * 0.6
    
    # L-shape should have exactly two adjacent sides filled
    # 如果恰好有两条相互垂直的边被填充，则认为是L形
    if (top_filled and left_filled and not bottom_filled and not right_filled) or \
       (top_filled and right_filled and not bottom_filled and not left_filled) or \
       (bottom_filled and left_filled and not top_filled and not right_filled) or \
       (bottom_filled and right_filled and not top_filled and not left_filled):
        return True
    
    # Check for L-pattern with corners
    corners = [
        (0, 0), (0, w-1), (h-1, 0), (h-1, w-1)
    ]
    
    filled_corners = sum(1 for y, x in corners if mask[y, x])
    
    # An L typically has one corner filled
    return filled_corners == 1

# 判断是否为十字
def is_cross(mask, height, width) -> bool:
    """
    Check if a mask represents a cross shape.
    Args:
        mask (np.ndarray): Binary mask of the object.
        height (int): Height of the bounding box.
        width (int): Width of the bounding box.
    Returns:
        bool: True if the mask is cross-shaped, False otherwise.
    """
    import numpy as np
    
    # Cross shape should have a central point with extensions in 4 directions
    if height < 3 or width < 3:
        return False
    
    # Check for symmetry around the center
    center_y, center_x = height // 2, width // 2
    
    # For a cross, the center point must be filled
    if not mask[center_y, center_x]:
        return False
    
    # Check for horizontal and vertical lines through center
    horizontal = np.sum(mask[center_y, :]) >= width * 0.6
    vertical = np.sum(mask[:, center_x]) >= height * 0.6
    
    return horizontal and vertical

# 找到相邻的对象
def find_adjacent_objects(objects, matrix_shape):
    """
    Find pairs of objects that are adjacent to each other.
    Args:
        objects (list): List of detected object dictionaries.
        matrix_shape (tuple): Shape of the original matrix.
    Returns:
        list: List of dictionaries describing adjacent object pairs.
    """
    import numpy as np
    
    adjacency_list = []
    
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects[i+1:], i+1):
            # 遍历每一对对象
            # 如果两个对象颜色相同，则跳过
            if obj1['color'] == obj2['color']:
                continue
                
            # Create full object masks
            mask1 = np.zeros(matrix_shape, dtype=bool)
            mask2 = np.zeros(matrix_shape, dtype=bool)
            
            # obj1和obj2的像素点变为True
            for y, x in obj1['pixels']:
                mask1[y, x] = True
                
            for y, x in obj2['pixels']:
                mask2[y, x] = True
            
            # Dilate mask1 to check for adjacency
            dilated = np.zeros(matrix_shape, dtype=bool)
            for y, x in obj1['pixels']:
                for dy, dx in [
                    (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal directions
                    (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal directions
                ]:
                    # OBJ1周围一圈的像素点的dilated变为True
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < matrix_shape[0] and 0 <= nx < matrix_shape[1]:
                        dilated[ny, nx] = True
            
            # Check if dilated mask1 overlaps with mask2
            # 如果上述dilated和obj2的掩码有重叠，则认为obj1和obj2相邻
            if np.any(np.logical_and(dilated, mask2)):
                adjacency_list.append({
                    'object1': i,
                    'object2': j,
                    'color1': obj1['color'],
                    'color2': obj2['color'],
                    'type1': obj1['type'],
                    'type2': obj2['type'],
                    'relationship': determine_spatial_relationship(obj1, obj2)
                })
    
    return adjacency_list

def determine_spatial_relationship(obj1, obj2):
    """
    Determine the spatial relationship between two objects.
    Args:
        obj1 (dict): First object dictionary.
        obj2 (dict): Second object dictionary.
    Returns:
        str: Description of the spatial relationship.
    """
    # Get the center points of both objects
    y1_min, x1_min = obj1['coordinates']['top_left']
    y1_max, x1_max = obj1['coordinates']['bottom_right']
    y2_min, x2_min = obj2['coordinates']['top_left']
    y2_max, x2_max = obj2['coordinates']['bottom_right']
    
    y1_center = (y1_min + y1_max) / 2
    x1_center = (x1_min + x1_max) / 2
    y2_center = (y2_min + y2_max) / 2
    x2_center = (x2_min + x2_max) / 2
    
    # Determine primary direction of object2 relative to object1
    # 计算两个对象的中心点之间的距离
    y_diff = y2_center - y1_center
    x_diff = x2_center - x1_center
    
    # Determine if objects are aligned (centers are aligned either horizontally or vertically)
    # 判断两个对象在水平、垂直方向上是否对齐（也就是中心点差距很小）
    h_aligned = abs(y_diff) < (obj1['size']['height'] + obj2['size']['height']) / 4
    v_aligned = abs(x_diff) < (obj1['size']['width'] + obj2['size']['width']) / 4
    
    if abs(y_diff) > abs(x_diff):
        if y_diff < 0:
            primary = "above"
        else:
            primary = "below"
    else:
        if x_diff < 0:
            primary = "to the left of"
        else:
            primary = "to the right of"
    
    # Check for special cases
    if h_aligned and primary in ["to the left of", "to the right of"]:
        return f"horizontally adjacent {primary}"  # 水平对齐
    
    if v_aligned and primary in ["above", "below"]:
        return f"vertically adjacent {primary}"  # 垂直对齐
    
    return primary

def get_overlap_description(percentage):
    """
    Generate a description of the overlap based on percentage.
    Args:
        percentage (float): Overlap percentage (0.0 to 1.0).
    Returns:
        str: Textual description of the overlap degree.
    """
    if percentage >= 0.9:
        return "almost completely overlapping"
    elif percentage >= 0.7:
        return "heavily overlapping"
    elif percentage >= 0.4:
        return "moderately overlapping"
    elif percentage >= 0.1:
        return "slightly overlapping"
    else:
        return "minimally overlapping"
    
def display_object_detection_results(results):
    """
    Display the detailed results of object detection with exact coordinates.
    Args:
        results (dict): The result dictionary from detect_objects().
    """
    myprint("\n=== OBJECT DETECTION RESULTS ===")
    myprint(f"Found {len(results['objects'])} objects:")
    
    # Display object details
    for obj in results['objects']:
        myprint(f"\nObject {obj['id']+1} - {obj['type']} of color {obj['color']}:")
        
        # Special handling for texture objects with multiple colors
        if obj['type'] == 'texture':
            myprint(f"  This is a texture containing colors: {obj['colors_present']}")
            myprint(f"  Color distribution: {obj['color_distribution']}")
        
        myprint(f"  Size: {obj['size']['description']} (area: {obj['size']['area']} pixels)")
        myprint(f"  Bounding Box: top={obj['bounding_box']['top']}, left={obj['bounding_box']['left']}, " +
              f"bottom={obj['bounding_box']['bottom']}, right={obj['bounding_box']['right']}")
        myprint(f"  Center Point: y={obj['coordinates']['center'][0]:.1f}, x={obj['coordinates']['center'][1]:.1f}")
    
    # Display adjacency information
    if results['adjacency']:
        myprint("\n=== ADJACENT OBJECTS ===")
        for adj in results['adjacency']:
            myprint(f"Object {adj['object1']+1} ({adj['type1']}) is {adj['relationship']} " +
                  f"Object {adj['object2']+1} ({adj['type2']})")
    
    # Display overlap information
    if results['overlaps']:
        myprint("\n=== OVERLAPPING OBJECTS ===")
        for overlap in results['overlaps']:
            myprint(f"Object {overlap['object1']+1} ({overlap['type1']}) and " +
                  f"Object {overlap['object2']+1} ({overlap['type2']}) are {overlap['description']}")
            myprint(f"  Overlap area: {overlap['overlap_area']} pixels " +
                  f"({overlap['overlap_percentage']:.1%} of smaller object)")
    
    myprint("\n=== OBJECT COUNT BY TYPE ===")
    for shape_type, count in results['counts'].items():
        myprint(f"  {shape_type}: {count}")
    
    myprint("==============================")

# Example usage (disabled to avoid side effects on import)
if False:
    matrix = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 2, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 4, 4, 4],
    [0, 0, 0, 0, 0, 0, 0, 4, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 4, 4, 4]
    ]

    # For testing overlapping objects
    overlap_test = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 2, 2, 0, 0],
    [0, 1, 1, 1, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 0],
    [0, 0, 0, 3, 3, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
    ]

    # For testing texture detection (no background)
    texture_test = [
    [1, 1, 1, 2, 2],
    [1, 3, 3, 2, 2],
    [4, 4, 3, 3, 5]
    ]

    myclear()
    results = detect_objects(matrix)
    display_object_detection_results(results)

    myprint("\n\n=== TESTING OVERLAP DETECTION ===")
    overlap_results = detect_objects(overlap_test)
    display_object_detection_results(overlap_results)

    myprint("\n\n=== TESTING TEXTURE DETECTION (NO BACKGROUND) ===")
    texture_results = detect_objects(texture_test)
    display_object_detection_results(texture_results)

    print(mymsg)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

fake_mode = not os.getenv('KAGGLE_IS_COMPETITION_RERUN')

import re
import time
import random
import warnings
from collections import Counter
import numpy as np, pandas as pd, polars as pl

import torch
import vllm
from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')
print('PyTorch version:', torch.__version__)
print('vLLM:', vllm.__version__)

def set_all_seeds(seed=GLOBAL_SEED):
    """
    Set all possible random seeds to ensure reproducibility across random, numpy, and torch.
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了完全确定性，禁用CUDA的非确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置Python哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds()

def extract_answer_from_r1(response) -> list:
    """
    Extract a matrix (list of lists of int) from a model response string using regex.
    Args:
        response (str): The response string containing a matrix in markdown format.
    Returns:
        list: The extracted matrix as a list of lists of integers, or an empty list if not found.
    """
    import re
    # Extract matrices like ```matrix\n1 2\n3 4\n```
    pattern = r"```matrix\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)
    
    # If matches found, take the last one
    if matches:
        matrix_str = matches[-1]  # Get the last match
        # Convert string matrix to 2D list
        matrix = []
        try:
            rows = matrix_str.strip().split('\n')
            for row in rows:
                if row.strip():  # Skip empty rows
                    matrix.append([int(cell) for cell in row.split()])
        except Exception as e:
            print(f"Error parsing matrix: {e}")
        return matrix
    else:
        return []
    
# Update the extraction function to extract Python code from the response
# 通过正则，提取convert函数
def extract_function_from_r1(response) -> str:
    """
    Extract a Python function named 'convert' from a model response string using regex.
    Args:
        response (str): The response string containing Python code.
    Returns:
        str: The extracted function code as a string, or an empty string if not found.
    """
    import re
    # ...existing code...

def extract_functions_from_r1(response) -> list:
    """
    Extract all Python functions named 'convert' from a model response string using regex.
    Args:
        response (str): The response string containing Python code.
    Returns:
        list: List of extracted function code strings.
    """
    import re
    # ...existing code...


# Function to execute the extracted code and get the result with improved error handling
def execute_function(function_str, input_matrix):
    """
    Execute a Python function string named 'convert' on the given input matrix with error handling and output validation.
    Args:
        function_str (str): The function code as a string.
        input_matrix (list): The input matrix to pass to the function.
    Returns:
        list: The result matrix, or an empty list if execution fails or output is invalid.
    """
    # ...existing code...
    
# Test the extraction function
# 测试提取函数
response = '''
</think>I need to analyze the pattern...

```python
def convert(input):
    # Create a copy of the input
    output = []
    for row in input:
        output.append(row.copy())
    
    # Perform a rotation of elements
    for i in range(len(output)):
        for j in range(len(output[0])):
            if output[i][j] > 0:
                output[i][j] = (output[i][j] + 1) % 10
                if output[i][j] == 0:
                    output[i][j] = 1
    
    return output
```
'''
print("Extracted function:", extract_function_from_r1(response))

def seed_everything(seed):
    """
    Set random seeds for reproducibility across random, numpy, and torch (with deterministic settings).
    Args:
        seed (int): The seed value to use.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
seed_everything(seed=0)

# Global placeholders set later in __main__
llm = None
tokenizer = None
cutoff_time = 0.0
llm_model_pth = '/kaggle/input/qwen-3/transformers/4b/1'
MAX_NUM_SEQS = 4
MAX_MODEL_LEN = 8196 * 3
arc_data = {}
submission = {}
training_solution = {}
evaluation_solution = {}
group_size = 0
predict_cnt = 0


import re
import keyword

from collections import Counter
import random

# Update the start_of_thinking prompt to explicitly require imports inside the function
# 更新 start_of_thinking 提示词，明确要求创建 'convert' 函数
start_of_thinking = "Okay, let's tackle this problem. I need to figure out the pattern from the training examples to create the 'convert' function. Let's look at the first training example. "

# Update the example format in the system prompt to request multiple solutions
# solution例子
eg_format = '''
After analyzing the pattern, I will create THREE DIFFERENT Python functions named 'convert' that transform the input matrix to the output matrix:

SOLUTION 1:
```python
def convert(input):
    #input: List[List[int]]
    #output: List[List[int]]
    # Import any necessary libraries INSIDE the function
    import numpy as np
    
    # First implementation approach
    # ...implementation details...
    
    return output
```
SOLUTION 2:
```python
def convert(input):
    #input: List[List[int]]
    #output: List[List[int]]
    # Import any necessary libraries INSIDE the function
    import numpy as np
    
    # Second implementation approach (different from Solution 1)
    # ...implementation details...
    
    return output
```
SOLUTION 3:
```python
def convert(input):
    #input: List[List[int]]
    #output: List[List[int]]
    # Import any necessary libraries INSIDE the function
    import numpy as np
    
    # Third implementation approach (different from both Solutions 1 and 2)
    # ...implementation details...
    
    return output
```
'''
# 系统提示词，包含角色，并要求创建3个不同的convert函数
system_prompt = "You are an expert at solving abstraction reasoning problems. User gives sample input matrices and sample output matrices. You need to learn the pattern and create THREE DIFFERENT Python functions that can transform the test input into the correct output.\n"
system_prompt += 'This is not a mathematical problem but a 2D visual reasoning problem where 0 usually represents the background, 1 to 9 represent different colors. Consider image processing techniques such as rotation, object detection, color substitution, color filling, object attraction, subgraph extraction, symmetry, gap filling, and so on.\n' 
system_prompt += 'IMPORTANT: You must name each function "convert" and place ALL IMPORTS INSIDE the function body, not outside. Do not include any code outside the function definitions.\n'
system_prompt += 'Make sure each of your three solutions takes a DIFFERENT APPROACH to solving the problem.\n' 
system_prompt += 'Your response should include THREE Python functions in this format: ' + eg_format

if fake_mode:
    arc_challenge_file = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json'
else:
    arc_challenge_file = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'

#arc_challenge_file = 'arc-agi_test_challenges.json'
import json

#载入arc_challenge_file

# 将list[list[int]]转换为字符串，以便发送给llm
def list2str(lst):
    """
    Convert a list of lists of integers to a string representation in markdown matrix format.
    Args:
        lst (list): List of lists of integers.
    Returns:
        str: String representation in markdown matrix format.
    """
    ans = '\n'.join([' '.join(map(str, sublist)) for sublist in lst])
    return '```matrix\n' + ans + '\n```'

sample_submission_path = '/kaggle/input/arc-prize-2025/sample_submission.json'

# 获得矩阵size，以及每种颜色出现的次数
def count_colors(matrix) -> str:
    """
    Count the occurrences of each color (0-9) in a matrix and return a formatted string.
    Args:
        matrix (list): List of lists of integers representing the matrix.
    Returns:
        str: String with matrix dimensions and color counts.
    """
    # ...existing code...

def get_case_llm_input(case_data):
    """
    Build the chat messages for a single ARC case, including:
        - Training examples with object detection summaries
        - All test inputs with object detection summaries
    Args:
        case_data (dict): Dictionary containing 'train' and 'test' data for the ARC case.
    Returns:
        list: List of message dictionaries for the LLM.
    """
    # Get training data
    train_data_list = case_data['train']

    # Build user prompt
    user_prompt = "Please Solve the below ARC(Abstraction reasoning contest) problems:\n"

    # Training examples
    for i, train_data in enumerate(train_data_list):
        myclear()
        results1 = detect_objects(train_data['input'])
        display_object_detection_results(results1)
        detection_info1 = mymsg

        myclear()
        results2 = detect_objects(train_data['output'])
        display_object_detection_results(results2)
        detection_info2 = mymsg

        input_count = count_colors(train_data['input'])
        output_count = count_colors(train_data['output'])

        user_prompt += (
            f"Training #{i+1} - Input:\n{list2str(train_data['input'])}\n"
            f"Input Stats: {input_count}\n{detection_info1}\n==============================\n"
            f"Output:\n{list2str(train_data['output'])}\n"
            f"Output Stats: {output_count}\n{detection_info2}\n==============================\n"
        )

    # Test inputs
    user_prompt += "Test Inputs:\n"
    for i, test_data in enumerate(case_data['test']):
        test_input_count = count_colors(test_data['input'])
        myclear()
        results = detect_objects(test_data['input'])
        display_object_detection_results(results)
        detection_info = mymsg
        user_prompt += (
            f"Test #{i+1} - Input:\n{list2str(test_data['input'])}\n"
            f"Input Stats: {test_input_count}\n{detection_info}\n==============================\n"
        )

    user_prompt += (
        "Your task is to provide THREE DIFFERENT implementations of the 'convert' function "
        "that can transform ALL test inputs correctly.\n"
    )

    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]

def llm_predict(batch_data):
    """
    Batch prediction function that extracts multiple solutions and applies them to all test inputs.
    Args:
        batch_data (list): List of tuples (case_id, message) for each ARC case.
    Returns:
        None. Updates the global submission dictionary.
    """
    # Check if we've reached the cutoff time
    if time.time() > cutoff_time:
        print("\n=== TIME LIMIT REACHED: Stopping prediction process ===")
        return
    # Generate model inputs
    model_inputs = [
        tokenizer.apply_chat_template(
            conversation=message,
            tokenize=False,
            add_generation_prompt=True,
        ) + start_of_thinking
        for case_id, message in batch_data
    ]

    # Configure sampling parameters with higher temperature to encourage diversity
    sampling_params = SamplingParams(
        temperature=0.5,            # Using higher temperature for diversity
        top_p=0.92,#核采样,只考虑累积概率达到92%
        min_p=0.05,#最小概率阈值
        skip_special_tokens=True,
        max_tokens=MAX_MODEL_LEN,
        # Request token-level logprobs to enable confidence estimation (if supported by vLLM version)
        logprobs=5,
    )
    
    # Execute prediction
    print("\n=== Generating multiple solution candidates per case ===")
    request_outputs = llm.generate(
        prompts=model_inputs,
        sampling_params=sampling_params,
    )
    
    # Helper: compute generation-level confidence (mean generated token logprob)
    def _gen_conf(req_out):
        try:
            # vLLM returns per-token logprobs in req_out.outputs[0].logprobs if requested
            out = req_out.outputs[0]
            # out.logprobs is a list[dict] or similar; collect chosen token logprobs when available
            if hasattr(out, 'logprobs') and out.logprobs:
                # Some versions store as list of dicts for each position containing top tokens; find chosen token logprob
                chosen_lps = []
                for pos, top_dict in enumerate(out.logprobs):
                    # If the structure is dict[token]->TokenProb, we need the chosen token id/text; fallback: mean of top
                    if isinstance(top_dict, dict) and len(top_dict) > 0:
                        # Take max logprob among top tokens as proxy
                        try:
                            best_lp = max(tp.logprob if hasattr(tp, 'logprob') else float(tp) for tp in top_dict.values())
                        except Exception:
                            # If values are floats
                            best_lp = max(float(v) for v in top_dict.values())
                        chosen_lps.append(best_lp)
                if chosen_lps:
                    return float(sum(chosen_lps) / len(chosen_lps))
            # Fallback: if cumulative logprob exists
            if hasattr(out, 'cumulative_logprob') and out.cumulative_logprob is not None:
                # Normalize by number of tokens if possible
                num_tokens = len(out.token_ids) if hasattr(out, 'token_ids') else 1
                return float(out.cumulative_logprob) / max(1, num_tokens)
        except Exception:
            pass
        return float('nan')

    # Helper: similarity between two matrices (exact cell-wise match ratio); returns 0 if shape differs
    def _matrix_similarity(a, b):
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

    # Helper: score a function candidate by how well it maps training inputs to outputs
    def _score_function_on_training(func_str, case_id):
        try:
            trains = arc_data[case_id].get('train', [])
            if not trains:
                return 0.0
            sims = []
            for pair in trains:
                pred = execute_function(func_str, pair['input'])
                sims.append(_matrix_similarity(pred, pair['output']))
            if not sims:
                return 0.0
            return float(sum(sims) / len(sims))
        except Exception:
            return 0.0

    # Process the results
    # 处理llm输出
    for i, output in enumerate(request_outputs):
        case_id = batch_data[i][0]
        prediction_text = output.outputs[0].text[:]
        gen_confidence = _gen_conf(output)
        
        # Extract all functions from the response
        # 提取所有函数
        function_candidates = extract_functions_from_r1(prediction_text)
        
        if fake_mode:
            print(model_inputs[i])
            print(prediction_text)
            print(f'\n====== CASE {case_id} - MULTIPLE SOLUTIONS ======')
            print(f'Found {len(function_candidates)} function candidates')
            if gen_confidence == gen_confidence:  # not NaN
                print(f"Generation confidence (mean logprob): {gen_confidence:.4f}")
        
        # Test each function against the first test input to check validity, and score on training set
        # 依次验证每个函数，并基于训练集相似度给出置信度打分
        valid_functions = []  # list of (func_str, preview_result, train_score)
        for idx, func_str in enumerate(function_candidates):
            if len(func_str) < 20:  # Skip very short functions which are likely invalid
                continue
                
            # Test on the first test input
            try:
                test_input = arc_data[case_id]['test'][0]['input']
                result = execute_function(func_str, test_input)
                if fake_mode:
                    print(func_str, result)
                
                if result and len(result) > 0:  # Only keep functions that produce valid output
                    train_score = _score_function_on_training(func_str, case_id)
                    valid_functions.append((func_str, result, train_score))
                    
                    if fake_mode:
                        print(f'Function candidate {idx+1} executed successfully; train_score={train_score:.3f}')
            except Exception as e:
                if fake_mode:
                    print(f'Function candidate {idx+1} failed: {str(e)}')

        if case_id not in submission:
            submission[case_id] = [dict(),dict(),dict()]
        # Process all test inputs with the valid functions
        # 验证函数后,对测试数据进行处理
        # Rank valid functions by training score (desc); break ties by longer function (heuristic) and generation confidence
        valid_functions.sort(key=lambda t: (t[2], len(t[0])), reverse=True)

        for test_idx, test_data in enumerate(arc_data[case_id]['test']):
            test_input = test_data['input']
            
            # Execute each valid function on this test input
            results = []  # list of (func_str, result, train_score)
            for func_str, _, train_score in valid_functions:
                try:
                    result = execute_function(func_str, test_input)
                    if result and len(result) > 0:
                        results.append((func_str, result, train_score))
                except Exception as e:
                    if fake_mode:
                        print(f'Error executing function on test #{test_idx+1}: {str(e)}')
            
            # 因为前面生成了3个函数,所以可能有多种结果,但正确答案只有一个
            # Select at most 2 different results for submission
            if len(results) >= 2:
                # Use the best-scored valid function as attempt_1
                func1, result1, score1 = results[0]
                submission[case_id][test_idx]['attempt_1'] = result1
                
                # Find a second function that produces a different result
                for func2, result2, score2 in results[1:]:
                    if result2 != result1:
                        submission[case_id][test_idx]['attempt_2'] = result2
                        break
                else:
                    # If all results are the same, use the second function anyway
                    func2, result2, _ = results[1]
                    submission[case_id][test_idx]['attempt_2'] = result2
            
            elif len(results) == 1:
                # If only one valid function, use it for both attempts
                func1, result1, _ = results[0]
                submission[case_id][test_idx]['attempt_1'] = result1
                submission[case_id][test_idx]['attempt_2'] = result1
            else:
                # No valid functions for this test input
                if fake_mode:
                    print(f'No valid functions for case {case_id}, test {test_idx}')
                submission[case_id][test_idx]['attempt_1'] = [[0, 0], [0, 0]]
                submission[case_id][test_idx]['attempt_2'] = [[0, 0], [0, 0]]
            
            # In fake mode, print additional debug information
            if fake_mode:
                correct_answer = None
                if case_id in training_solution and len(training_solution[case_id]) > test_idx:
                    correct_answer = training_solution[case_id][test_idx]
                elif case_id in evaluation_solution and len(evaluation_solution[case_id]) > test_idx:
                    correct_answer = evaluation_solution[case_id][test_idx]
                
                if correct_answer:
                    match1 = submission[case_id][test_idx]['attempt_1'] == correct_answer
                    match2 = submission[case_id][test_idx]['attempt_2'] == correct_answer
                    print(f'\n【Test #{test_idx+1} correct_answer】:{correct_answer}')
                    print(f"【attempt_1】:{submission[case_id][test_idx]['attempt_1']}")
                    print(f"【attempt_2】:{submission[case_id][test_idx]['attempt_2']}")
                    print(f"【result1】:{'correct' if match1 else 'wrong'}")
                    print(f"【result2】:{'correct' if match2 else 'wrong'}")
                    print(f"【overall】:{'SUCCESS' if (match1 or match2) else 'FAILURE'}  | gen_conf={gen_confidence if gen_confidence==gen_confidence else 'NA'}")

    
def alter_zero(submission):
    """
    Check all case_ids in the submission. If attempt_1 or attempt_2 is empty or [[0]],
    replace with [[0,0],[0,0]].
    Args:
        submission (dict): The submission result dictionary.
    Returns:
        dict: The modified submission dictionary.
    """
    # 创建一个2x2的零矩阵作为替代值
    replacement = [[0, 0], [0, 0]]
    
    # 遍历所有case_id
    for case_id in submission:
        # 遍历每个case_id的所有测试
        for test_idx in range(len(submission[case_id])):
            if 'attempt_1' not in submission[case_id][test_idx]:
                submission[case_id][test_idx]['attempt_1'] = replacement
                submission[case_id][test_idx]['attempt_2'] = replacement
            # 检查attempt_1
            attempt_1 = submission[case_id][test_idx]['attempt_1']
            if not attempt_1 or (len(attempt_1) == 1 and len(attempt_1[0]) == 1 and attempt_1[0][0] == 0):
                submission[case_id][test_idx]['attempt_1'] = replacement
                
            # 检查attempt_2
            attempt_2 = submission[case_id][test_idx]['attempt_2']
            if not attempt_2 or (len(attempt_2) == 1 and len(attempt_2[0]) == 1 and attempt_2[0][0] == 0):
                submission[case_id][test_idx]['attempt_2'] = replacement
    
    return submission

if __name__ == "__main__":
    """
    Main script execution for ARC solver. Loads data, runs LLM predictions, and saves submission.
    """
    start_time = time.time()
    cutoff_time = start_time + (11 * 60 + 30) * 60

    llm = LLM(
        llm_model_pth,
        dtype="bfloat16",
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        seed=2024,
    )
    tokenizer = llm.get_tokenizer()

    with open(arc_challenge_file, 'r') as f:
        arc_data = json.load(f)

    group_size = 4
    predict_cnt = 8 if fake_mode else 1000

    with open(sample_submission_path, 'r') as f:
        submission = json.load(f)

    training_solution_path = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'
    with open(training_solution_path, 'r') as f:
        training_solution = json.load(f)
    evaluation_solution_path = '/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json'
    with open(evaluation_solution_path, 'r') as f:
        evaluation_solution = json.load(f)

    batch_data = []
    for case_id in arc_data:
        predict_cnt -= 1
        if predict_cnt < 0:
            break
        case_llm_input = get_case_llm_input(arc_data[case_id])
        batch_data.append((case_id, case_llm_input))
        if len(batch_data) >= group_size:
            llm_predict(batch_data)
            batch_data = []
    if len(batch_data) > 0:
        llm_predict(batch_data)

    submission = alter_zero(submission)
    submission_path = 'submission.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f)
    print(f"Submission saved to {submission_path}")