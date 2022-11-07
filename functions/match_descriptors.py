import numpy as np
import itertools

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    ssd_single = lambda patch1, patch2: np.sum((patch1-patch2)**2)
    distances = np.zeros((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            distances[i][j] = ssd_single(desc1[i], desc2[j])
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1   
        matches = np.array([[i, np.argmin(distances[i])] for i in range(len(desc1))])        
    elif method == "mutual":
        forth = np.array([[i, np.argmin(distances[i])] for i in range(len(desc1))])
        back = np.array([[i, np.argmin(distances.T[i])] for i in range(len(desc2))])
        matches = np.array([[i, forth[i][1]] for i in range(len(forth)) if (back[forth[i][1]][1] == i)])
    elif method == "ratio":
         matches = np.array([[i, np.argmin(distances[i])] for i in range(len(desc1))
                             if 2 * np.min(distances[i]) < np.partition(distances[i], 1)[1]])        
    else:
        matches = np.array([])
    return matches




