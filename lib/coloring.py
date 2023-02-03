# colored Module
from functools import singledispatchmethod
from typing import Iterable, Optional, Self, TypeAlias

import numpy as np
import numpy.typing as npt
from termcolor import colored

Color: TypeAlias = str
OnColor: TypeAlias = str
Attrs: TypeAlias = Iterable[str]


class Coloring:
    """wrapper class of termcolor.colored

    Returns:
        _type_: Coloring
    """

    __is_color_monitor: bool = True

    @staticmethod
    def __csv_to_list(csv: Optional[str]) -> list[str] | None:
        if csv is None:
            return csv
        return csv.split(',')

    @staticmethod
    def __iterable_to_csv(lst: Iterable[str] | None) -> str | None:
        """iterable of string to csv string"""
        if lst is None:
            return None
        return ','.join(lst)

    # colored
    @singledispatchmethod
    @classmethod
    def coloring(
        cls,
        text: str,
        color: Optional[Color] = None,
        on_color: Optional[OnColor] = None,
        attrs: Optional[Iterable[str] | str] = None,
    ) -> str:
        """set color escape sequences to string,
        overloaded by list[str] or NDArray[str]"""
        if not cls.__is_color_monitor:
            return text
        if isinstance(attrs, str):
            if ',' in attrs:
                attrs = cls.__csv_to_list(attrs)
            else:
                attrs = [attrs]
        return colored(text, color, on_color=on_color, attrs=attrs)

    @coloring.register(list)
    @classmethod
    def _(
        cls,
        lst: list[str],
        color: Optional[Color] = None,
        on_color: Optional[OnColor] = None,
        attrs: Optional[Attrs] = None,
    ) -> str:
        """set color escape sequences to string for list (overload)"""
        return np.vectorize(cls.coloring)(
            lst, color, on_color, cls.__iterable_to_csv(attrs)
        )

    @coloring.register(np.ndarray)
    @classmethod
    def _(
        cls,
        arr: npt.NDArray[np.str_],
        color: Optional[Color] = None,
        on_color: Optional[OnColor] = None,
        attrs: Optional[Attrs] = None,
    ) -> str:
        """set color escape sequence to string for ndarray (overload)"""
        out = np.vectorize(cls.coloring)(
            arr, color, on_color, cls.__iterable_to_csv(attrs)
        )
        return out

    # colored_mask
    @classmethod
    def coloring_mask(
        cls,
        arr: npt.NDArray[np.str_],
        mask: npt.NDArray[np.bool_],
        color: Optional[Color] = None,
        on_color: Optional[OnColor] = None,
        attrs: Optional[Attrs] = None,
    ) -> npt.NDArray[np.str_]:
        """set color escape sequence to string in ndarray according to boolean array"""
        out = arr.astype(object)
        np.putmask(out, mask, cls.coloring(out, color, on_color, attrs))
        return out

    def __new__(cls, is_color_monitor: bool = True) -> Self:  # type: ignore
        self = super().__new__(cls)
        cls.__is_color_monitor = is_color_monitor
        return self


if __name__ == '__main__':
    # from repr import Repr
    Coloring(True)

    str1_1: str = '0010'
    str1_2: str = '0010'
    list1: list[str] = ['1101', '0010']
    ndarray1: npt.NDArray[np.str_] = np.array([['1101', '0010'], ['0010', '1101']])

    str2: str = 'ABC'
    list2: list[str] = ['A', 'B', 'C']
    ndarray2: npt.NDArray[np.str_] = np.array([['A', 'B', 'C'], ['D', 'E', 'F']])
    ndarray3: npt.NDArray[np.str_] = np.array([['A', 'X', 'C'], ['Y', 'E', 'Z']])
    mask3: npt.NDArray[np.bool_] = ndarray2 == ndarray3

    print('colored:')
    print(Coloring.coloring(str1_1, 'blue'))
    print(Coloring.coloring(str1_2, 'yellow'))
    print(Coloring.coloring(list1, 'cyan'))
    print(Coloring.coloring(ndarray1, 'red', attrs=['bold', 'underline']))

    print(Coloring.coloring(str2, 'blue', attrs=['bold', 'underline']))
    print(Coloring.coloring(list2, 'cyan'))
    print(Coloring.coloring(ndarray2, 'red'))

    print(Coloring.coloring(str2, attrs=['bold', 'underline']))
    print(Coloring.coloring(list2, attrs=['bold']))
    print(Coloring.coloring(ndarray2, attrs=['bold', 'underline']))

    print('colored_mask:')
    print(Coloring.coloring_mask(ndarray2, mask3, 'yellow'))
    print(Coloring.coloring_mask(ndarray2, mask3, 'yellow', attrs=['underline']))
    print(Coloring.coloring_mask(ndarray2, mask3, 'yellow', attrs=['bold', 'underline']))

    # print(Repr.repr(Coloring.coloring(str1_1, 'blue')))
