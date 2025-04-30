
def path_to_links(path: list) -> list:
    """
    Converts a given path to a list of tuples containing two elements each.

    Args:
        path (list): A list of elements representing a path.

    Returns:
        list: A list of tuples, each tuple containing two elements from the given path.
    """
    if len(path) == 1:
        return [(path[0], path[0])]
    return [(path[i], path[i+1]) for i in range(len(path)-1)]
