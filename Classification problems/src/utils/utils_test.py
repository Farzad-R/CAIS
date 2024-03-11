import numpy as np


def utils_test_function(arg1: np.ndarray, arg2: np.ndarray, arg3: np.ndarray) -> np.ndarray:
    """
    Perform element-wise multiplication between arg2 and arg3, then add the result to arg1.

    Parameters:
    arg1 (np.ndarray): The first input array.
    arg2 (np.ndarray): The second input array.
    arg3 (np.ndarray): The third input array.

    Returns:
    np.ndarray: The resulting array after performing the element-wise operations.
    """
    return arg1 + (arg2 * arg3)


# Create some sample arrays
arg1 = np.array([1, 2, 3])
arg2 = np.array([4, 5, 6])
arg3 = np.array([7, 8, 9])

# Call the function
result = utils_test_function(arg1, arg2, arg3)
print(result)
