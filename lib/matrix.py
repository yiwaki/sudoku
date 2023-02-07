# Matrix Class, Block SuperClass - Row SubClass, Column SubClass, Square SubClass
from __future__ import annotations

from abc import abstractmethod
from typing import Self, Type, TypeAlias

import numpy as np
import numpy.typing as npt

Shape: TypeAlias = tuple[int, int]  # (height, width)

BlockNo: TypeAlias = np.int_
BlockPos: TypeAlias = np.int_

Location: TypeAlias = tuple[int, int]  # (block_no, position)

BlockLayout: TypeAlias = npt.NDArray[BlockNo]
PositionLayout: TypeAlias = npt.NDArray[BlockPos]


class Matrix:
    """Matrix super class"""

    # Type Alias
    Binary: TypeAlias = npt.NDArray[np.int16]
    Decimal: TypeAlias = npt.NDArray[np.str_]
    Mask: TypeAlias = npt.NDArray[np.bool_]

    Address: TypeAlias = tuple[int, int]  # (row, column)

    # Member Variables
    type: str = 'Matrix'
    size: int = 9
    shape: Shape = (size, size)

    @classmethod
    def cell_no_to_addr(cls, cell_no: int) -> Address:
        return (cell_no // cls.size, cell_no % cls.size)

    def __new__(cls) -> Self:  # type: ignore
        pass


class Block:
    """Block class

    Args:
        Matrix (object): Matrix superclass

    Returns:
        Self: Block
    """

    # member variables
    type: str = 'Block'
    shape: Shape
    block_layout: BlockLayout
    position_layout: PositionLayout

    types: set[Type[Row] | Type[Column] | Type[Square]] = set()

    @classmethod
    def block_mask(cls, block_no: BlockNo) -> Matrix.Mask:
        """get mask array of block no. of the block type

        Args:
            block_no (BlockNo): block no

        Returns:
            Matrix.Mask: boolean array to indicate elements belong to block no
        """
        return cls.block_layout == block_no

    @classmethod
    def location_to_block_mask(cls, location: Location) -> Matrix.Mask:
        """create boolean array indicating elements belong to location

        Args:
            location (Location): tuple[block_no, position]

        Returns:
            Matrix.Mask: boolean array indicating elements belong to location
        """
        block_mask: Matrix.Mask = cls.block_layout == location[0]
        pos_mask: Matrix.Mask = cls.position_layout == location[1]
        return block_mask & pos_mask

    @classmethod
    def cell_count_in_block(cls) -> int:
        """returns count of cells in block

        Returns:
            int: count of cells in block
        """
        return cls.shape[0] * cls.shape[1]

    @classmethod
    def addr_to_loc(cls, addr: Matrix.Address) -> Location:
        """transfer address to location of block type

        Args:
            addr (Matrix.Address): tuple[row, column]

        Returns:
            Location: tuple[block_no, position]
        """
        return (cls.block_layout[addr], cls.position_layout[addr])

    @classmethod
    @abstractmethod
    def to_row_no_s(cls, block_no: BlockNo) -> list[BlockNo]:
        ...

    @classmethod
    @abstractmethod
    def to_column_no_s(cls, block_no: BlockNo) -> list[BlockNo]:
        ...

    @classmethod
    @abstractmethod
    def to_square_no_s(cls, block_no: BlockNo) -> list[BlockNo]:
        ...

    def __new__(cls) -> Self:  # type: ignore
        return super().__new__(cls)


# Row SubClass
class Row(Block):
    """Row Structure"""

    type: str = 'ROW'
    shape: Shape = (1, Matrix.size)
    block_layout: BlockLayout
    position_layout: PositionLayout

    @classmethod
    def to_row_no_s(cls, row_no: BlockNo) -> list[BlockNo]:
        return [row_no]

    @classmethod
    def to_column_no_s(cls, row_no: BlockNo) -> list[BlockNo]:
        return [BlockNo(i) for i in range(Matrix.size)]

    @classmethod
    def to_square_no_s(cls, row_no: BlockNo) -> list[BlockNo]:
        return [BlockNo(i) for i in range(row_no // 3 * 3, row_no // 3 * 3 + 3)]

    def __new__(cls) -> Self:  # type: ignore
        """constructor"""
        self = super().__new__(cls)

        cls.block_layout = np.array(
            [n // (cls.shape[0] * cls.shape[1]) for n in range(Matrix.size**2)]
        ).reshape(Matrix.shape)
        cls.position_layout = np.array(
            [n % (cls.shape[0] * cls.shape[1]) for n in range(Matrix.size**2)]
        ).reshape(Matrix.shape)

        super().types.add(cls)
        return self


class Column(Block):
    """Column class

    Args:
        Block (object): Block super class

    Returns:
        Self: Column
    """

    type: str = 'COLUMN'
    shape: Shape = (Matrix.size, 1)
    block_layout: BlockLayout
    position_layout: PositionLayout

    @classmethod
    def to_row_no_s(cls, column_no: BlockNo) -> list[BlockNo]:
        return [BlockNo(i) for i in range(Matrix.size)]

    @classmethod
    def to_column_no_s(cls, column_no: BlockNo) -> list[BlockNo]:
        return [column_no]

    @classmethod
    def to_square_no_s(cls, column_no: BlockNo) -> list[BlockNo]:
        return [BlockNo(i) for i in range(column_no % 3, Matrix.size, 3)]

    def __new__(cls) -> Self:  # type: ignore
        """constructor"""
        self = super().__new__(cls)

        cls.block_layout = np.array(
            [n % (cls.shape[0] * cls.shape[1]) for n in range(Matrix.size**2)]
        ).reshape(Matrix.shape)

        cls.position_layout = np.array(
            [n // (cls.shape[0] * cls.shape[1]) for n in range(Matrix.size**2)]
        ).reshape(Matrix.shape)

        super().types.add(cls)
        return self


class Square(Block):
    """Square class

    Args:
        Block (_type_): Block super class

    Returns:
        Self: Square
    """

    type: str = 'SQUARE'
    shape: Shape = (Matrix.size // 3, Matrix.size // 3)
    block_layout: BlockLayout
    position_layout: PositionLayout

    @classmethod
    def to_row_no_s(cls, square_no: BlockNo) -> list[BlockNo]:
        return [BlockNo(i) for i in range(square_no // 3 * 3, square_no // 3 * 3 + 3)]

    @classmethod
    def to_column_no_s(cls, square_no: BlockNo) -> list[BlockNo]:
        return [BlockNo(i) for i in range(square_no % 3 * 3, square_no % 3 * 3 + 3)]

    @classmethod
    def to_square_no_s(cls, square_no: BlockNo) -> list[BlockNo]:
        return [square_no]

    def __new__(cls) -> Self:  # type: ignore
        """constructor"""
        self = super().__new__(cls)

        tmp1_1: npt.NDArray[np.int_]
        for b in range(0, 9, 3):
            tmp1_2: npt.NDArray[np.int_] = np.concatenate(
                [
                    np.full(cls.shape, b),
                    np.full(cls.shape, b + 1),
                    np.full(cls.shape, b + 2),
                ],
                axis=1,
            )
            if b == 0:
                tmp1_1 = tmp1_2
            else:
                tmp1_1 = np.concatenate([tmp1_1, tmp1_2], axis=0)
        cls.block_layout = tmp1_1

        tmp2: PositionLayout
        tmp2 = np.arange(cls.shape[0] * cls.shape[1]).reshape(cls.shape)
        tmp2 = np.concatenate([tmp2, tmp2, tmp2], axis=0)
        tmp2 = np.concatenate([tmp2, tmp2, tmp2], axis=1)
        cls.position_layout = tmp2

        super().types.add(cls)

        return self


if __name__ == '__main__':
    print('##################################################')
    print(Matrix.cell_no_to_addr(7))
    print(Matrix.cell_no_to_addr(80))
    print(Matrix.cell_no_to_addr(56))

    block_no = BlockNo(3)
    print('##################################################')
    Row()
    Row()
    Row()
    print(f'{Row.type=}')
    print(f'{Row.shape=}')
    print(f'{Row.block_layout=}')
    print(f'{Row.position_layout=}')
    print(f'{Row.block_mask(block_no)=}')
    print(f'{Row.cell_count_in_block()=}')
    print(f'{Row.addr_to_loc((2, 5))=}')
    for i in range(1, 9, 3):
        print(Row.to_row_no_s(BlockNo(i)))
        print(Row.to_column_no_s(BlockNo(i)))
        print(Row.to_square_no_s(BlockNo(i)))

    print('##################################################')
    Column()
    Column()
    Column()
    print(f'{Column.type=}')
    print(f'{Column.shape=}')
    print(f'{Column.block_layout=}')
    print(f'{Column.position_layout=}')
    print(f'{Column.block_mask(block_no)=}')
    print(f'{Column.cell_count_in_block()=}')
    print(f'{Column.addr_to_loc((2, 5))=}')
    for i in range(1, 9, 3):
        print(f'{Column.to_row_no_s(BlockNo(i))=}')
        print(f'{Column.to_column_no_s(BlockNo(i))=}')
        print(f'{Column.to_square_no_s(BlockNo(i))=}')

    print('##################################################')
    Square()
    Square()
    Square()
    print(f'{Square.type=}')
    print(f'{Square.shape=}')
    print(f'{Square.block_layout=}')
    print(f'{Square.position_layout=}')
    print(f'{Square.block_mask(block_no)=}')
    print(f'{Square.cell_count_in_block()=}')
    print(f'{Square.addr_to_loc((2, 5))=}')
    for i in range(1, 9, 3):
        print(f'{Square.to_row_no_s(BlockNo(i))=}')
        print(f'{Square.to_column_no_s(BlockNo(i))=}')
        print(f'{Square.to_square_no_s(BlockNo(i))=}')

    print('##################################################')
    for block in Block.types:
        print(f'{block.type=}')
