# -*- coding: utf-8 -*-
"""
Copyright Fabien Mathieu
fabien.mathieu@normalesup.org

This file is part of GISMO.
"""


class MyClass3:
    """A whatever-you-are-doing.

    Parameters
    ----------
    a : float
        The `a` of the system.
    b : float
        The `b` of the system.

    Examples
    --------
        >>> my_object = MyClass3(a = 5, b = 3)
    """

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def addition(self) -> float:
        """Add `a` and `b`.

        Returns
        -------
        Float
            The sum of `a` and `b`.

        Examples
        --------
            >>> my_object = MyClass3(a=5, b=3)
            >>> my_object.addition()
            8
        """
        return self.a + self.b
