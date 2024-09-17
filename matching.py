from tqdm import tqdm
import numpy as np
import joblib
from scipy.cluster.vq import vq
from numpy.linalg import norm
import cv2
from typing import List, Tuple, Dict

all_descriptors = np.load("output/all_descriptors.npy", allow_pickle=True)
keypoints = np.load("output/all_points.npy", allow_pickle=True)
img_size = np.load("output/img_size.npy", allow_pickle=True)
k, codebook = joblib.load("output/bow_codebook.plk")

# import torch
# from lightglue import LightGlue
# from lightglue.utils import rbd
# torch.set_grad_enabled(False)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# matcher = LightGlue(features='disk').eval().to(device)

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

print("Build Pairs")

visual_words = []

for desciptors in all_descriptors:
    img_visual_words, distance = vq(desciptors.astype("float"), codebook)
    visual_words.append(img_visual_words)

frequency_vectors = []
for img_visual_words in visual_words:
    img_frequency_vector = np.zeros(k)
    for word in img_visual_words:
        img_frequency_vector[word] += 1
    frequency_vectors.append(img_frequency_vector)

frequency_vectors = np.stack(frequency_vectors)

N = frequency_vectors.shape[0]

df = np.sum(frequency_vectors > 0, axis = 0 )

idf = np.log(N/ df)

tfidf = frequency_vectors * idf

all_idx = []
all_score = []
top_k = 10
for i in range(N):
    a = tfidf[i]
    b = tfidf
    cosine_similarity = np.dot(a, b.T)/(norm(a) * norm(b, axis=1))
    idx = np.argsort(-cosine_similarity)[1:top_k].tolist()
    score = np.sort(-cosine_similarity)[1:top_k]
    all_idx.append(idx)
    all_score.append(score)

matching_indices = [None]*N

for i in range(N):
    for j, id in enumerate(all_idx[i]):
        if not matching_indices[i]:
            matching_indices[i] = []
        if not matching_indices[id]:
            matching_indices[id] = []
        if -all_score[i][j] > 0.75:
            if not id in matching_indices[i]:
                matching_indices[i].append(id)
            if not i in matching_indices[id]:
                matching_indices[id].append(i)

print(matching_indices)

from collections import defaultdict

def find_largest_connected_with_smallest_score(N, matching_lists):
    """
    Find the best score graph with largest number of connected components.
    
    Args:
    N: Number of images
    matching_lists: List of arrays of matching image indices for each image
    
    Returns:
    Tuple containing:
    - Best graph with connected components
    - Best score of the graph
    """

    # Step 1: Create graph representation
    graph = defaultdict(list)
    for i, matches in enumerate(matching_lists):
        for j, node in enumerate(matches):
            graph[i].append((node, j + 1))  # j + 1 is the score

    # Step 2: Find all connected components
    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor, _ in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)

    visited = [False] * N
    components = []
    for node in range(N):
        if not visited[node]:
            component = []
            dfs(node, component)
            components.append(component)

    # Step 3: Calculate score for each component
    def calculate_score(component):
        score = 0
        max_score = 0
        for node in component:
            for neighbor, s in graph[node]:
                if s > max_score:
                    max_score = s
                if neighbor in component:
                    score += s
        return score, max_score  # No need to divide by 2 now

    # Step 4: Find largest component with smallest score
    best_component = None
    best_score = float('inf')
    best_size = 0
    best_max_component_score = 0

    for component in components:
        score, max_component_score = calculate_score(component)
        size = len(component)
        if size > best_size or (size == best_size and score < best_score):
            best_component = component
            best_score = score
            best_size = size
            best_max_component_score = max_component_score

    return best_component, best_score, best_max_component_score

image_sequence, score, max_component_score = find_largest_connected_with_smallest_score(N, matching_indices)
print(f"Largest connected component with smallest score: {image_sequence}")
print(f"Score: {score}")
print(f"Max component score: {max_component_score}")
                
#                 # feats0 = {
#                 #     "keypoints": torch.tensor(np.array([keypoints[reference_id]], dtype=float), dtype=torch.float).to(device), 
#                 #     "descriptors": torch.tensor(np.array([all_descriptors[reference_id]], dtype=float), dtype=torch.float).to(device), 
#                 #     'image_size': torch.tensor(np.array([img_size[reference_id]], dtype=float), dtype=torch.float).to(device)
#                 # }
#                 # feats1 = {
#                 #     "keypoints": torch.tensor(np.array([keypoints[id]], dtype=float), dtype=torch.float).to(device), 
#                 #     "descriptors": torch.tensor(np.array([all_descriptors[id]], dtype=float), dtype=torch.float).to(device), 
#                 #     'image_size': torch.tensor(np.array([img_size[id]], dtype=float), dtype=torch.float).to(device)
#                 # }

#                 # matches01 = matcher({'image0': feats0, 'image1': feats1})
#                 # feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
#                 # kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
#                 # m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
#                 # idx0, idx1 = matches[..., 0].detach().cpu().numpy(), matches[..., 1].detach().cpu().numpy()

def reconstruct_3d_points(
    image_sequence: List[int],
    descriptors: List[np.ndarray],
    keypoints: List[np.ndarray],
    matching_indices: List[List[int]]
) -> Tuple[List[np.ndarray], List[List[Tuple[int, np.ndarray]]]]:
    """
    Reconstruct 3D points from a sequence of images and their associated keypoints.
    
    Args:
    image_sequence: Sequence of image indices
    descriptors: List of keypoint's descriptors
    keypoints: List of arrays of keypoints for each image
    matching_indices: List of arrays of matching image indices for each image
    
    Returns:
    Tuple containing:
    - List of 3D points
    - List of corresponding 2D keypoints for each 3D point
    """
    
    # Initialize data structures
    points_3d = []
    points_2d_projections = []
    point_to_keypoint = {}  # Dictionary to map keypoints to 3D points
    keypoint_to_3d_point = {}
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Iterate through the sequence A
    for current_image_index in tqdm(image_sequence):
        matching_image_indices = matching_indices[current_image_index]
        
        for reference_image_index in matching_image_indices[:max_component_score]:
            current_descriptors = descriptors[current_image_index].astype(np.float32)
            reference_descriptors = descriptors[reference_image_index].astype(np.float32)

            # Match descriptors using FLANN
            matches = flann.knnMatch(current_descriptors, reference_descriptors, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            # Process good matches
            for match in good_matches:
                current_kp = match.queryIdx
                reference_kp = match.trainIdx
                
                # Check if the reference keypoint is already associated with a 3D point
                if (current_image_index, current_kp) in keypoint_to_3d_point and \
                    (reference_image_index, reference_kp) in keypoint_to_3d_point:
                    continue
                elif (reference_image_index, reference_kp) in keypoint_to_3d_point:
                    # Add the current keypoint to the existing 3D point
                    point_3d_index = keypoint_to_3d_point[(reference_image_index,  reference_kp)]
                    # Check if the current image index is not already in the projection list
                    if not (current_image_index, point_3d_index) in point_to_keypoint:
                        points_2d_projections[point_3d_index].append((current_image_index, current_kp))
                        keypoint_to_3d_point[(current_image_index, current_kp)] = point_3d_index
                        point_to_keypoint[(current_image_index, point_3d_index)] = current_kp
                elif (current_image_index, current_kp) in keypoint_to_3d_point:
                    # Add the current keypoint to the existing 3D point
                    point_3d_index = keypoint_to_3d_point[(current_image_index, current_kp)]
                    # Check if the current image index is not already in the projection list
                    if not (current_image_index, point_3d_index) in point_to_keypoint:
                        points_2d_projections[point_3d_index].append((current_image_index, current_kp))
                        keypoint_to_3d_point[(reference_image_index, reference_kp)] = point_3d_index
                        point_to_keypoint[(reference_image_index, point_3d_index)] = reference_kp
                else:
                    # Create a new 3D point
                    new_3d_point = np.random.rand(3)  # Placeholder for actual 3D reconstruction
                    points_3d.append(new_3d_point)
                    point_3d_index = len(points_3d) - 1
                    
                    # Store 2D projections
                    points_2d_projections.append([
                        (current_image_index, current_kp),
                        (reference_image_index, reference_kp)
                    ])
                    
                    # Update the keypoint_to_3d_point dictionary
                    keypoint_to_3d_point[(current_image_index, current_kp)] = point_3d_index
                    keypoint_to_3d_point[(reference_image_index, reference_kp)] = point_3d_index
                    point_to_keypoint[(current_image_index, point_3d_index)] = current_kp
                    point_to_keypoint[(reference_image_index, point_3d_index)] = reference_kp
    
    return points_3d, points_2d_projections

reconstructed_3d_points, corresponding_2d_keypoints = reconstruct_3d_points(image_sequence, all_descriptors, keypoints, matching_indices)

print(f"Number of reconstructed 3D points: {len(reconstructed_3d_points)}")
print(f"Example 3D point: {reconstructed_3d_points[0]}")
print(f"Corresponding 2D keypoints for the first 3D point: {corresponding_2d_keypoints[0]}")

# print(queue, len(queue))
# np.save('output/img_pairs.npy', queue[1:])
# np.save('output/all_matches.npy', np.array(all_matches, dtype=object))
