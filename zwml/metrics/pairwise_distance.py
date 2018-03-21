import numpy as np
import pandas as pd

def pandas_to_numpy(x):
    """
    Checks if the input is a Dataframe or series, converts to numpy matrix for
    calculation purposes.
    ---
    Input: X (array, dataframe, or series)
    Output: X (array)
    """
    if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
        return x.as_matrix()
    if type(x) == type(np.array([1,2])):
        return x
    return np.array(x) 

def manhattan_distance(vec1, vec2):
    """
    Manhattan distance measures the distance along
    each direction and sums them together.
    """
    vec1 = pandas_to_numpy(vec1)
    vec2 = pandas_to_numpy(vec2)
    return np.sum(np.abs(vec1-vec2))

def euclidean_distance(vec1, vec2):
    """
    Calculating the Euclidean distance which is
    the more traditional method for distance 
    calculation. sqrt((x1-x2)^2 + (y1-y2)^2 + ...)
    """
    vec1 = pandas_to_numpy(vec1)
    vec2 = pandas_to_numpy(vec2)
    return np.sqrt(np.sum((vec1-vec2)**2))

def cosine_similarity_without_numpy(vec1, vec2):
    """
    Calculates the angular similarity of two vectors.
    Does so by calculating cos(theta) between the vectors
    using the dot product.
    
    cos_sim = A dot B/(magnitude(A)*magnitude(B))
    """
    dot_product=0
    vec1_sum_sq = 0
    vec2_sum_sq = 0
    for idx, val in enumerate(vec1):
        dot_product += val*vec2[idx]
        vec1_sum_sq += val*val
        vec2_sum_sq += vec2[idx]*vec2[idx]
    return dot_product/(vec1_sum_sq**0.5*vec2_sum_sq**0.5)

def cosine_similarity(vec1,vec2):
    """
    Calculates the angular similarity of two vectors.
    Does so by calculating cos(theta) between the vectors
    using the dot product.
    
    cos_sim = A dot B/(magnitude(A)*magnitude(B))
    """
    vec1 = pandas_to_numpy(vec1)
    vec2 = pandas_to_numpy(vec2)
    dot_product = np.dot(vec1, vec2)
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    return dot_product/(vec1_norm* vec2_norm)

def gaussian_kernel(vec1, vec2, bandwidth=1.):
    """
    Returns the Gaussian kernel relationship between two
    vectors. The Gaussian kernel assumes a bandwidth that
    defines the "width" of the Gaussian used to determine
    the relationship between the two points.
    """
    dist = euclidean_distance(vec1, vec2)
    norm = 1/(np.sqrt(2*np.pi*bandwidth**2))
    return norm*np.exp(-dist**2/(2*bandwidth**2))

def uniform_kernel(vec1, vec2, threshold_range=1, value=0.5):
    """
    Returns a value if the two provided vectors are
    within threshold range of each other. In normal
    implementation, the integration of value over the
    whole range should be 1.
    """
    distance = euclidean_distance(vec1, vec2)
    if distance <= threshold_range:
        probs = value
    else:
        probs = 0.
    return probs

def rbf_kernel(vec1, vec2, gamma=None):
    """
    The RBF, or radial basis function, kernel
    is similar to the gaussian kernel. However,
    it has a different scaling factor, using
    gamma instead of the bandwidth for normalization
    and width scaling. Gamma defaults to 1/dimensions
    unless otherwise specified.d
    """
    if not gamma:
        gamma = 1/len(vec1)
    distance = euclidean_distance(vec1, vec2)**2
    distance *= -gamma
    return np.exp(distance)