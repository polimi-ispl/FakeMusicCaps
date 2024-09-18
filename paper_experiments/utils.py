import  random

def normalize_tensor(tensor):
    # Ensure tensor is in float format
    tensor = tensor.float()

    # Compute min and max values
    min_val = tensor.min()
    max_val = tensor.max()

    # Normalize to the range [0, 1]
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    # Scale to range [-1, 1]
    normalized_tensor = normalized_tensor * 2 - 1

    return normalized_tensor


def split_list(lst, ratio=0.8):
    """
    Splits the input list into two disjoint sets based on the given ratio.

    Args:
        lst (list): The list to split.
        ratio (float): The ratio for the first subset (default is 0.8 for 80%).

    Returns:
        tuple: A tuple containing two lists. The first list contains the ratio percentage of the original list, and the second list contains the remaining percentage.
    """
    # Calculate the number of elements for the first subset
    n = len(lst)
    split_index = int(n * ratio)

    # Shuffle the list to ensure randomness
    shuffled_lst = lst[:]
    random.shuffle(shuffled_lst)

    # Split the shuffled list into two parts
    subset1 = shuffled_lst[:split_index]
    subset2 = shuffled_lst[split_index:]

    return subset1, subset2
