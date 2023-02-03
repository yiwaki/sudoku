# Bitmap, BitmapArray Class
from functools import singledispatchmethod
from typing import Optional, Self

import numpy as np
import numpy.typing as npt


class Bitmap:
    """toolbox of bitmap

    Returns:
        _type_: Bitmap
    """

    digit: int = 16
    full_bits: np.int_ = np.int_(2**digit - 1)

    # to_binary (overloaded by argument of list or NDArray)
    @singledispatchmethod
    @classmethod
    def to_binary(cls, val: int) -> str:
        """integer to binary string. overloaded by list and npt.NDArray

        Args:
            val (int | list[int] | ndarray[int]): bitmap integer.

        Returns:
            str | list[str] | ndarray[str]: binary string.
        """
        if not isinstance(val, int):
            # for int in zero dimension NDArray
            val = int(val)
        return np.binary_repr(val, width=cls.digit)

        """"""

    @to_binary.register(list)  # type: ignore
    @classmethod
    def _(cls, lst: list[int]) -> list[str]:
        """integer to binary string for list (overload)"""
        if len(lst) == 0:
            return []
        return (np.vectorize(cls.to_binary))(lst)

    @to_binary.register(np.ndarray)  # type: ignore
    @classmethod
    def _(cls, arr: npt.NDArray[np.int_]) -> npt.NDArray[np.str_]:
        """integer to binary string for ndarray (overload)"""
        return (np.vectorize(cls.to_binary))(arr)

    # popcount (overloaded by argument of list or NDArray)
    @singledispatchmethod
    @classmethod
    def popcount(cls, val: int) -> int:
        """count of bits on bitmap. overloaded by list and npt.NDArray.

        Args:
            val (int | list[int] | ndarray[numpy.int_]): bitmap integer

        Returns:
            int | list[int] | ndarray[numpy.int_]: bit count of the bitmap integer.
        """
        return bin(val).count('1')

    @popcount.register(list)  # type: ignore
    @classmethod
    def _(cls, lst: list[int]) -> list[int]:
        """count of bits on bitmaps in list (overload)"""
        return np.vectorize(cls.popcount)(lst)

    @popcount.register(np.ndarray)  # type: ignore
    @classmethod
    def _(cls, arr: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """count of bits on bitmaps in array (overload)"""
        return np.vectorize(cls.popcount)(arr)

    # min_popcount (arguments of NDArray only)
    @classmethod
    def min_popcount(
        cls,
        arr: list[int] | npt.NDArray[np.int_],
        mask: Optional[npt.NDArray[np.bool_]] = None,
    ) -> npt.NDArray[np.bool_]:
        """returns boolean array indicating elements where the smallest popcount exists.

        Args:
            arr (list[int] | ndarray[np.int_]): bitmap integers
            mask (npt.NDArray[np.bool_]] | None): boolean array
            indicating the calculation target. Defaults to None.

        Returns:
            ndarray[np.bool_]: boolean array indicating elements where
            the smallest popcount exists
        """
        arr_tmp: npt.NDArray[np.int_] = np.array(arr, dtype=np.int_)
        pop_cnt: npt.NDArray[np.int_]
        if mask is None:
            pop_cnt = cls.popcount(arr_tmp)  # type: ignore
            return pop_cnt == np.amin(pop_cnt)
        else:
            pop_cnt = cls.popcount(arr_tmp)  # type: ignore
            tmp = pop_cnt == np.amin(pop_cnt[mask])
            return tmp & mask

    # rightmost_bit (overloaded by argument of list and NDArray)
    @singledispatchmethod
    @classmethod
    def rightmost_bit(cls, val: np.int_) -> np.int_:
        """leftmost bit of integer. overloaded by list and npt.NDArray.

        Args:
            val (np.int_ | list[int] | ndarray[numpy.int_]): bitmap integer

        Returns:
            np.int_ | list[int] | ndarray[numpy.int_]: leftmost bit integer.
        """
        for i in range(cls.digit):
            lmb: np.int_ = np.int_(1 << i)
            if val & lmb != 0:
                return lmb
        return np.int_(0)

    @rightmost_bit.register(list)  # type: ignore
    @classmethod
    def _(cls, lst: list[int]) -> list[int]:
        """leftmost bit of integer for list (overload)"""
        return np.vectorize(cls.rightmost_bit)(lst)

    @rightmost_bit.register(np.ndarray)  # type: ignore
    @classmethod
    def _(cls, arr: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """leftmost bit of integer for ndarray (overload)"""
        return np.vectorize(cls.rightmost_bit)(arr)

    @singledispatchmethod
    @classmethod
    def bmp_to_dec(cls, val: np.int_) -> str:
        """integer to decimal string

        Args:
            val (np.int_ | list[int] | ndarray[np.int_]): bitmap integer.

        Returns:
            str | list[str] | ndarray[numpy.str_]: decimal string
        """
        decimals = list(cls.to_binary(val))
        decimals.reverse()
        return ''.join([str(n + 1) for n, v in enumerate(decimals) if v == '1'])

    @bmp_to_dec.register(list)  # type: ignore
    @classmethod
    def _(cls, lst: list[int]) -> list[str]:
        """integer to decimal string for list (overload)"""
        return np.vectorize(cls.bmp_to_dec)(lst)

    @bmp_to_dec.register(np.ndarray)  # type: ignore
    @classmethod
    def _(cls, arr: npt.NDArray[np.int_]) -> npt.NDArray[np.str_]:
        """integer to decimal string for ndarray (overload)"""
        return np.vectorize(cls.bmp_to_dec)(arr)

    @classmethod
    def split_single_bit(cls, val: int) -> list[int]:
        """split integer to list of each single bit integer

        Args:
            val (int): bitmap integer

        Returns:
            list[int]: list of each splitted single bit integer
        """
        out: list[int] = []
        for n in range(cls.digit, -1, -1):
            if int(val) & (1 << n):
                out.append(1 << n)
        return out

    def __new__(cls, digit: int) -> Self:  # type: ignore
        """constructor of Bitmap class

        Args:
            digit (int): digit of bitmap

        Returns:
            Self: Bitmap class object
        """
        self = super().__new__(cls)
        cls.digit = digit
        cls.full_bits = 2**cls.digit - 1
        return self


if __name__ == '__main__':
    Bitmap(4)
    print(Bitmap.to_binary('123'))
    print(Bitmap.to_binary(Bitmap.full_bits))

    bin1: int = 0b1001
    list1: list[int] = [0b1111, 0b0100]
    narr1: npt.NDArray[np.int_] = np.array([[0b1010, 0b1000], [0b0100, 0b1111]])

    print(f'{type(bin1)=}')
    print(f'{type(list1)=}')
    print(f'{type(narr1)=}')

    bin2: int = 0b1010
    list2: list[int] = [0b1011, 0b0110, 0b0001]
    narr2: npt.NDArray[np.int_] = np.array([[0b0010, 0b1010], [0b0010, 0b1101]])

    print('to_binary:')
    print(f'bin1={Bitmap.to_binary(bin1)}')
    print(f'list1={Bitmap.to_binary(list1)}')
    print(f'narr1{Bitmap.to_binary(narr1)}')

    print('-' * 10)
    print('popcount:')
    print(f'bin1={Bitmap.popcount(bin1)}')
    print(f'list1={Bitmap.popcount(list1)}')
    print(f'narr1={Bitmap.popcount(narr1)}')

    print('-' * 10)
    print('min_popcount:')
    print(f'list1={Bitmap.min_popcount(list1)}')
    print(f'narr1={Bitmap.min_popcount(narr1)}')

    print('-' * 10)
    print('min_popcount(mask):')
    print(f'{narr1 > 8}')
    print(f'{narr1 >= 8}')
    print(f'{Bitmap.min_popcount(narr1, narr1 >= 8)}')

    print('-' * 10)
    print('leftmost_bit:')
    print(f'bin1={Bitmap.to_binary(Bitmap.rightmost_bit(bin1))}')
    print(f'list1={Bitmap.to_binary(Bitmap.rightmost_bit(list1))}')
    print(f'narr1={Bitmap.to_binary(Bitmap.rightmost_bit(narr1))}')

    print('-' * 10)
    print('bmp_to_dec:')
    print(Bitmap.bmp_to_dec(bin1))
    print(Bitmap.bmp_to_dec(list1))
    print(Bitmap.bmp_to_dec(narr1))

    print('-' * 10)
    print('split_single_bit:')
    print(Bitmap.split_single_bit(bin1))
