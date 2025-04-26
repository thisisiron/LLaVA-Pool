def badly_formatted_function(x, y):
    """This function has some style issues.

    Args:
        x: First parameter
        y: Second parameter
    """
    if x > y:
        return x - y
    else:
        result = x + y
        return result


# Some unnecessary imports


def another_function():
    """This docstring uses single quotes instead of double quotes."""
    values = [1, 2, 3, 4]
    for i in values:
        print(i)
