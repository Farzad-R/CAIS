
class src_class():
    """
    This class represents a source class.

    Attributes:
        arg1 (int): The first argument.
        arg2 (int): The second argument.
    """

    def __init__(self) -> None:
        """
        Initialize the source class with default arguments.
        """
        self.arg1 = 1
        self.arg2 = 2

    def src_method_1(self, arg1: int, arg2: int) -> int:
        """
        Add two integers together.

        Args:
            arg1 (int): The first integer.
            arg2 (int): The second integer.

        Returns:
            int: The sum of arg1 and arg2.
        """
        return arg1 + arg2
