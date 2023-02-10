# Repr Class
import re
from functools import singledispatchmethod
from typing import Self

import numpy as np
import numpy.typing as npt
from tabulate import tabulate


class Repr_:
    """represent literals"""

    ESC: str = '\x1b'
    OB: str = '\['

    @staticmethod
    def _rem_color_esc(txt: str) -> str:
        """remove color escape sequences from string"""
        return re.sub(f'{Repr_.ESC}{Repr_.OB}[0-9]+m', '', txt)

    @singledispatchmethod
    @classmethod
    def repr(cls, arr: list) -> str:
        """create string to represent literals of list.
        overloaded by int, str and numpy.array

        Args:
            arr (list): literal of list

        Returns:
            str: strings for print function
        """
        return cls.repr(np.array(arr, dtype=np.str_))

    @repr.register(np.ndarray)
    @classmethod
    def _(cls, arr: np.ndarray) -> str:
        """create string to represent literals of numpy.ndarray (overload)"""
        if arr.size == 1:
            return cls.repr(str(arr))

        if arr.ndim == 1:
            out: str = ''
            for i in [len(cls._rem_color_esc(n)) for n in arr]:
                out += '-' * i + ' '
            out += '\n'
            out += ' '.join(arr) + '\n'
            for i in [len(cls._rem_color_esc(n)) for n in arr]:
                out += '-' * i + ' '
            return out
        else:
            return tabulate(arr, tablefmt='simple')

    @repr.register(int)
    @classmethod
    def _(cls, i: int) -> str:
        """create string to represent integer (overload)"""
        return cls.repr(str(i))

    @repr.register(str)
    @classmethod
    def _(cls, s: str) -> str:
        """create string to represent string (overload)"""
        out: str = '-' * len(cls._rem_color_esc(s)) + '\n'
        out += s + '\n'
        out += '-' * len(cls._rem_color_esc(s))
        return out

    def __new__(cls) -> Self:  # type: ignore
        return super().__new__(cls)


if __name__ == '__main__':
    print(Repr_.repr('test'))
    lis1: list[str] = ['test1', 'test11', 'test333']
    lis2: list[str] = ['test22', 'test222', 'test3']
    lis3: list[str] = ['test333', 'test2', 'test33']
    print(Repr_.repr(lis1))
    ndarr1: npt.NDArray[np.str_] = np.array(lis1)
    print(Repr_.repr(ndarr1))
    ndarr2: npt.NDArray = np.array([lis1, lis2, lis3])
    print(Repr_.repr(ndarr2))

    print(Repr_.repr(100))
    ilis1: list[int] = [1, 11, 333]
    ilis2: list[int] = [22, 222, 3]
    ilis3: list[int] = [33, 2, 33]
    inarr: npt.NDArray[np.int_] = np.array([ilis1, ilis2, ilis3])
    print(Repr_.repr(inarr))

    xlis = [1, 'a', 333.33]
    print(Repr_.repr(xlis))
