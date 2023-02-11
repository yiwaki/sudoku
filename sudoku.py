import argparse
import logging
import os
import time
from enum import Enum
from typing import Optional, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from IPython.display import display
from pandas.io.formats.style import Styler
from tabulate import tabulate

import lib.matrix as mx
from lib import dfstyle
from lib.bitmap import Bitmap
from lib.coloring import Coloring
from lib.log import Log
from lib.matrix import Block, Column, Matrix, Row, Square
from lib.repr import Repr_
from sudoku import c_brute_force  # type: ignore


# Log Control Class
class LogCtrl:
    """control log message"""

    _logger: Log
    _handler_name: Log.HandlerName = 'logfile'
    _log_filepath: Log.FilePath

    __msg_level: dict[str, int] = {
        'D': logging.DEBUG,
        'I': logging.INFO,
        'W': logging.WARNING,
        'E': logging.ERROR,
        'C': logging.CRITICAL,
        'N': logging.NOTSET,
    }
    __msg_area_level: dict[str, int] = {
        'PRUNE_BITS': logging.INFO,
        'ISOLATED_BITS': logging.INFO,
        'BRUTE_FORCE': logging.INFO,
        'DEFAULT': logging.INFO,
    }

    def into_msg_area(self, msg_area_name: str = 'DEFAULT') -> None:
        self._logger.setLevel(self.__msg_area_level[msg_area_name])
        msg = f'changed message level({msg_area_name}): '
        msg += f'{self.__msg_area_level[msg_area_name]}'
        self._logger.info(msg)

    def quiet(self, True_=True):
        if True_:
            Log.resetHandler(self._logger)
        else:
            Log.setHandler(self._logger, self._handler_name, self._log_filepath)

    def msg_level(self, msg_lv: str = '') -> Self:
        i_msg = 'set message level:'
        lv_len = len(self.__msg_area_level)
        llv = list((str.upper(msg_lv) + 'N' * lv_len)[0:lv_len])
        for n, lv in enumerate(llv):
            area_name = [n for n in self.__msg_area_level.keys()][n]
            self.__msg_area_level[area_name] = self.__msg_level[lv]
        self.into_msg_area()
        i_msg += f'MSG_AREA_LEVEL: {self.__msg_area_level}'
        self._logger.info(i_msg)
        return self

    def __init__(self) -> None:
        pass


# Input/Output Class
class IO_:
    """input/output"""

    result: Matrix.Decimal

    _problem_filepath: str
    _problem: Matrix.Binary
    _working: Matrix.Binary
    _trial: int
    _handler_name: Log.HandlerName
    _log_filepath: str
    _given_no: int

    __STY_PROBLEM: dict[str, str] = {'color': 'blue', 'font-weight': 'bold'}
    __STY_RESULT: dict[str, str] = {}

    def __prepare(self) -> None:
        """prepare initial bitmap"""
        self._trial = 0
        self._working = (lambda x: 1 << (x - 1))(self._problem)
        self._working = np.where(self._working == 0, Bitmap.full_bits, self._working)
        self._given_no = np.where(self._working != Bitmap.full_bits, 1, 0).sum()

    def stats(self, matrix_: Optional[Matrix.Binary] = None) -> str:
        """status message"""
        if matrix_ is None:
            matrix_ = self._working
        solved = np.count_nonzero(Bitmap.popcount(matrix_) == 1)
        out = f'\n{solved - self._given_no} cells solved, '
        out += f'total {solved} cells'
        return out

    def __style_table(self, style: Styler) -> Styler:
        return style.set_table_styles(
            [{'selector': 'th.row_heading', 'props': [('font-weight', 'bold')]}]
        )

    def display_result(self) -> None:
        """display result (styler)"""
        df = pd.DataFrame(self.result)
        style = df.style.set_caption(f'<<< result : {self._problem_filepath} >>>')
        mask: Matrix.Mask = self.result == self.problem

        style = dfstyle.set_property_mask(style, mask, self.__STY_PROBLEM)
        style = dfstyle.set_property_mask(style, ~mask, self.__STY_RESULT)
        style = self.__style_table(style)
        display(style)

    def display_result_text(self) -> None:
        """display result (text)"""
        print(f'<<< result : {self._problem_filepath} >>>')
        mask: Matrix.Mask = self.result == self.problem

        tmp = Coloring.coloring_mask(self.result, mask, 'blue', attrs=['bold'])
        out = tabulate(
            tmp,
            tablefmt='simple',
            headers=range(9),  # type: ignore
            showindex=True,
            stralign='center',
            numalign='center',
        )
        print(out, end='')
        print(self.stats())

    def __load_csv(self, filepath: str) -> bool:
        """load sudoku problem from csv file"""
        logger_name = os.path.splitext(os.path.basename(filepath))[0]
        self._logger = Log.getLogger(logger_name)
        Log.setHandler(self._logger, self._handler_name)
        self._problem_filepath = filepath

        self._problem = np.loadtxt(filepath, delimiter=',').astype(np.uint16)
        if self._problem.shape != Matrix.shape:
            print('unexpected data')
            return False

        self.problem = np.where(self._problem == 0, '', self._problem)
        self._working = np.empty((9, 9), dtype=np.uint16)

        return True

    def __load_text(self, filepath: str) -> bool:
        """load sudoku problem from file"""
        dd: str | None = None
        with open(filepath, 'r') as f:
            dd = f.read()
        logger_name = os.path.splitext(os.path.basename(filepath))[0]
        self._logger = Log.getLogger(logger_name)
        Log.setHandler(self._logger, self._handler_name)
        self._problem_filepath = filepath

        tmp = dd.replace('\n', '').replace(' ', '').replace('\t', '')
        if len(tmp) == Matrix.size**2:
            self._problem = np.array(list(tmp), dtype=np.uint16).reshape(
                Matrix.shape
            )  # type: ignore
            self.problem = np.where(self._problem == 0, '', self._problem)
            self._working = np.empty(Matrix.shape, dtype=np.uint16)
        else:
            print('unexpected data')
            return False

        return True

    def load(self, filepath: str) -> Self:
        if not os.path.exists(filepath):
            print(f'{filepath} is not found')
            return self

        status: bool
        ext = os.path.splitext(os.path.basename(filepath))[1]
        if ext == '.csv':
            status = self.__load_csv(filepath)
        else:
            status = self.__load_text(filepath)

        if not status:
            print(f'failed to load: {filepath}')

        self.__prepare()
        return self

    def display_problem(self) -> None:
        """display problem (styler)"""
        df = pd.DataFrame(np.where(self._problem == 0, '', self._problem))
        style = df.style.set_caption(f'<<< sudoku : {self._problem_filepath} >>>')
        mask = self._problem != 0
        style = dfstyle.set_property_mask(style, mask, self.__STY_PROBLEM)
        style = self.__style_table(style)
        display(style)

    def display_problem_text(self) -> None:
        """display problem (text)"""
        print(f'<<< sudoku : {self._problem_filepath} >>>')
        tmp: Matrix.Decimal = np.where(self._problem == 0, '', self._problem)
        tmp = Coloring.coloring(tmp, 'blue', attrs=['bold'])  # type: ignore
        out = tabulate(
            tmp,
            tablefmt='simple',
            headers=range(9),  # type: ignore
            showindex=True,
            stralign='center',
            numalign='center',
        )
        print(out)

    @staticmethod
    def log_color(color_: bool = True):
        Coloring(color_)

    def __init__(self) -> None:
        Coloring(True)


# Verify Class
class Verify:
    _working: Matrix.Binary
    _problem_filepath: str

    def __verify_missing_bits(self) -> bool:
        """whether there ara any missing bits"""
        for block in Block.types:
            for block_no in range(block.cell_count_in_block()):
                mask: Matrix.Mask = block.block_mask(mx.BlockNo(block_no))
                if np.bitwise_or.reduce(self._working[mask]) != Bitmap.full_bits:
                    return False
        return True

    def __verify_cells_complete(self) -> bool:
        """Whether all cells are uniquely complete"""
        popcount_array: npt.NDArray[np.int_] = Bitmap.popcount(self._working)  # type: ignore
        return np.all(popcount_array == 1)  # type: ignore

    def verify(self) -> None:
        """whether the result is correct"""
        print(f'verify : {self._problem_filepath} >>> ', end='')
        status: bool = True
        if not self.__verify_cells_complete():
            print('incomplete, ', end='')
            status = False

        if not self.__verify_missing_bits():
            print('missing numbers, ', end='')
            status = False

        if status:
            print('successfully verified', end='')

        print()

    def __init__(self) -> None:
        pass


# ProneBits（ビット剪定）Class
STAR_CNT: int = 5


class PruneBits(IO_):
    """Prone Bits (ビット剪定) algorism"""

    _working: Matrix.Binary
    _trial: int
    _bf_cnt: int
    _logger: Log

    _popcount_sum: dict[object, list[int]]

    def __log_working(self) -> None:
        """log of self._working for debug"""
        mask: Matrix.Mask = Bitmap.popcount(self._working) == 1  # type: ignore
        tmp1: Matrix.Decimal = Bitmap.to_binary(self._working)  # type: ignore
        tmp2 = Coloring.coloring_mask(tmp1, mask, 'yellow')

        d_msg = f'\n{Repr_.repr(tmp2)}'
        d_msg += self.stats()
        self._logger.debug(d_msg)

    def __prune_bits_each(self, block: Row | Column | Square) -> bool:
        """execute prune bits algorism"""
        i_msg = '*' * STAR_CNT + f' prune bits ({block.type}): '
        i_msg += f'trial #{self._trial}-{self._bf_cnt} ' + '*' * STAR_CNT
        self._logger.info(i_msg)

        change: bool = False
        self.__log_working()
        for block_no in range(Matrix.size):
            mask = block.block_mask(mx.BlockNo(block_no))
            self._popcount_sum[block][block_no] = Bitmap.popcount(
                self._working[mask]
            ).sum()  # type: ignore

            if self._popcount_sum[block][block_no] == Matrix.size:
                self._logger.debug(f'{block.type}: #{block_no}: >>>> skip')
                continue

            popcount_before: int = self._popcount_sum[block][block_no]

            self._logger.debug(f'{block.type}: #{block_no}')

            solved_mask = Bitmap.popcount(self._working) == 1
            unsolved_mask = np.logical_not(solved_mask)
            solved_bmp = np.bitwise_or.reduce(self._working[mask & solved_mask])

            np.putmask(self._working, mask & unsolved_mask, self._working & ~solved_bmp)

            self._popcount_sum[block][block_no] = Bitmap.popcount(
                self._working[mask]
            ).sum()  # type: ignore

            if self._popcount_sum[block][block_no] != popcount_before:
                change = True

            self.__log_working()

        return change

    def prune_bits(self) -> tuple[bool, bool]:
        """returns boolean tuple (finished, changed)"""
        i_msg = '*' * STAR_CNT
        i_msg += f' brute force : trial #{self._trial} ' + '*' * STAR_CNT
        self._logger.info(i_msg)

        finished: bool = False
        changed: bool = True
        while changed:
            changed = False
            for block in Block.types:
                changed = self.__prune_bits_each(block) or changed  # type: ignore
                finished = sum(self._popcount_sum[block]) == Matrix.size**2
                if finished:
                    return (finished, changed)

        i_msg = f'prune bits : trial#{self._trial} over'
        self._logger.info(i_msg)

        return (finished, changed)

    def __init__(self) -> None:
        Block()  # type: ignore
        Row()
        Column()
        Square()


# IsolatedBit Class
class IsolatedBit(IO_):
    """A bit that doesn't exist in the same blocks (Row, Column and Square)
    is called an isolated bit.
    """

    _working: Matrix.Binary
    _logger: Log
    _trial: int
    _bf_cnt: int

    _popcount_sum: dict[object, list[int]]

    def __is_isolated_on_other_blocks(
        self,
        addr: Matrix.Address,
        iso_bit: np.uint16,
        omitted_block: Row | Column | Square,
    ) -> bool:
        """True if the same bits don't exist on the other cells in
        the same block. In case of True, the bit is called 'isolated'.
        """
        i_msg = f'isolated {Bitmap.to_binary(iso_bit)} in {addr}'
        self._logger.info(i_msg)

        block_types = Block.types.copy()
        block_types.remove(omitted_block)  # type: ignore
        for block in block_types:
            d_msg = f'check by {block.type}'

            block_no, pos = block.addr_to_loc(addr)
            mask = block.block_mask(mx.BlockNo(block_no))
            mask[addr] = False
            oth = np.bitwise_or.reduce(self._working[mask], axis=None)

            d_msg += f'\nall bits of the other cells on {block.type}: '
            d_msg += f'{Bitmap.to_binary(oth)}'
            self._logger.debug(d_msg)

            if oth == Bitmap.full_bits:
                return False  # not isolated
        return True  # isolated

    def __log_isolated_case(
        self, unsolved_mask: Matrix.Mask, addr: Matrix.Address, iso_bit: np.uint16
    ) -> None:
        d_msg = f'\n  {Bitmap.to_binary(self._working[addr])} {addr}'
        tmp = self._working[unsolved_mask]
        oth = Coloring.coloring(
            f'-){Bitmap.to_binary(np.bitwise_or.reduce(tmp))}', attrs=['underline']
        )
        d_msg += '\n' + oth

        c_iso_bit = Coloring.coloring(Bitmap.to_binary(iso_bit), attrs=['bold'])
        d_msg += f'\n  {c_iso_bit}'
        self._logger.debug(d_msg)
        self._logger.info(f'{addr} <- {Bitmap.bmp_to_dec(iso_bit)}')

        tmp1: Matrix.Decimal = Bitmap.to_binary(self._working)  # type: ignore
        mask: Matrix.Mask = Bitmap.popcount(self._working) == 1  # type: ignore
        tmp2 = Coloring.coloring_mask(tmp1, mask, 'yellow')
        d_msg = f'\n{Repr_.repr(tmp2)}'
        d_msg += self.stats()
        self._logger.debug(d_msg)

    def __log_not_isolated_case(
        self, unsolved_mask: Matrix.Mask, addr: Matrix.Address, candidate: np.uint16
    ) -> None:
        tmp = Bitmap.to_binary(self._working[addr])
        d_msg = f'\n  {tmp} : {addr}'
        bmp = np.bitwise_or.reduce(self._working[unsolved_mask])
        oth = Coloring.coloring(f'-){Bitmap.to_binary(bmp)}', attrs=['underline'])
        d_msg += '\n' + oth
        c_check = Bitmap.to_binary(candidate)
        d_msg += f'\n  {c_check}'
        d_msg += f'\n{addr} cannot change'

    def __isolated_bits_each(self, block: Row | Column | Square) -> bool:
        """pickup isolated bits in three blocks"""
        d_msg = '*' * STAR_CNT + f' pickup isolated bits ({block.type}): '
        d_msg += f'after trial #{self._trial} ' + '*' * STAR_CNT
        change_count: int = 0
        for block_no in range(Matrix.size):
            d_msg = f'{block.type}: block#{block_no}'
            self._logger.debug(d_msg)
            mask = block.block_mask(mx.BlockNo(block_no))
            unsolved_mask = np.logical_and(Bitmap.popcount(self._working) != 1, mask)

            self._popcount_sum[block][block_no] = Bitmap.popcount(
                self._working[mask]
            ).sum()  # type: ignore

            popcount_before: int = self._popcount_sum[block][block_no]

            tmp = np.where(unsolved_mask)
            tars = [(tmp[0][i], tmp[1][i]) for i in range(len(tmp[0]))]
            d_msg = f'unsolved cells: {tars}'
            self._logger.debug(d_msg)
            for addr in tars:
                unsolved_mask[addr] = False
                mask_bmp: np.uint16 = np.bitwise_or.reduce(self._working[unsolved_mask], axis=None)
                candidate: np.uint16 = self._working[addr] & ~mask_bmp

                d_msg = ''
                if Bitmap.popcount(candidate) == 1:
                    if not self.__is_isolated_on_other_blocks(addr, candidate, block):
                        d_msg += f'\nskipped: not isolate bit on blocks including {addr}'
                        continue

                    self._working[addr] = candidate
                    self.__log_isolated_case(unsolved_mask, addr, candidate)

                else:
                    unsolved_mask[addr] = True
                    self.__log_not_isolated_case(unsolved_mask, addr, candidate)

            self._popcount_sum[block][block_no] = Bitmap.popcount(
                self._working[mask]
            ).sum()  # type: ignore

            if self._popcount_sum[block][block_no] != popcount_before:
                change_count += popcount_before - self._popcount_sum[block][block_no]

        if change_count == 0:
            self._logger.info('no isolated bit found')
            return False  # changed
        else:
            self._logger.info(f'{change_count} isolated bits found and set')
            return True  # not changed

    def isolated_bits(self) -> tuple[bool, bool]:
        """returns boolean tuple (finished, changed)"""
        i_msg = '*' * STAR_CNT + ' pickup isolated bits : '
        i_msg += 'after trial #{self._trial} ' + '*' * STAR_CNT
        # self._logger.info(i_msg)

        i_msg = '*' * STAR_CNT + ' pickup isolated bits '
        i_msg += f'after brute force #{self._trial} ' + '*' * STAR_CNT
        self._logger.info(i_msg)

        finished: bool = False
        changed: bool = True
        while changed:
            for block in Block.types:
                changed = self.__isolated_bits_each(block) and changed  # type: ignore

            finished = sum(self._popcount_sum[Row]) == Matrix.size**2
            if finished:
                return (finished, changed)

        return (finished, changed)

    def __init__(self) -> None:
        pass


# BruteForce（総当たり）Class


class BruteForce(IO_):
    _logger: Log

    def __log_matrix(self, matrix_: Matrix.Binary) -> None:
        mask: Matrix.Mask = Bitmap.popcount(matrix_) == 1  # type: ignore
        tmp1: Matrix.Decimal = Bitmap.to_binary(matrix_)  # type: ignore
        tmp2 = Coloring.coloring_mask(tmp1, mask, 'yellow')

        d_msg = f'\n{Repr_.repr(tmp2)}'
        d_msg += self.stats(matrix_)
        self._logger.debug(d_msg)

    def __valid(self, matrix_: Matrix.Binary, mask: Matrix.Mask) -> bool:
        """whether the changes are valid"""
        status: bool = True
        if np.any(Bitmap.popcount(matrix_[mask]) == 0):
            self._logger.debug('popcount==0')
            status = False

        coverage: np.uint16 = np.bitwise_or.reduce(matrix_[mask])
        if coverage != Bitmap.full_bits:
            self._logger.debug(f'missing bits coverage={Bitmap.to_binary(coverage)}')
            status = False
        return status

    def __done(self, matrix_: Matrix.Binary) -> bool:
        """whether the result is correct"""
        for block in Block.types:
            for block_no in range(block.cell_count_in_block()):
                mask: Matrix.Mask = block.block_mask(mx.BlockNo(block_no))
                coverage: np.uint16 = np.bitwise_or.reduce(matrix_[mask])
                if coverage != Bitmap.full_bits:
                    self._logger.debug(
                        f'{block.type} {block_no=} ' f'coverage={Bitmap.to_binary(coverage)}'
                    )
                    return False

        status: bool = np.all(Bitmap.popcount(matrix_) == 1)  # type: ignore
        if not status:
            self._logger.debug('popcount != 0')
        return status

    def prune_by_pivot(
        self, matrix_: Matrix.Binary, pivot: mx.Location, bit: np.uint16
    ) -> Optional[Matrix.Binary]:
        matrix_work: Matrix.Binary = matrix_.copy()
        pivot_bmp: str = Bitmap.to_binary(matrix_work[pivot])
        target_bit: str = Bitmap.to_binary(bit)
        self._logger.debug(f'{pivot=} {pivot_bmp=} {target_bit=}')

        matrix_work[pivot] = bit
        for block in Block.types:
            block_no: mx.BlockNo
            position: mx.BlockPos
            block_no, position = block.addr_to_loc(pivot)  # type: ignore
            self._logger.debug(f'{block.type}: {block_no=}')

            mask: Matrix.Mask = block.block_mask(block_no)
            mask[pivot] = False
            np.putmask(matrix_work, mask, matrix_work & ~np.uint16(bit))
            mask[pivot] = True

            if not self.__valid(matrix_work, mask):
                self._logger.debug('invalid')
                return None

        return matrix_work

    def brute_force(self, matrix_: Matrix.Binary, cell_no: int = -1) -> Matrix.Binary:
        """run brute force algorism"""
        cell_no += 1
        if cell_no == 81:
            # end of cell
            self._logger.debug('reached end of cell')
            return matrix_

        addr = Matrix.cell_no_to_addr(cell_no)
        matrix_work: Matrix.Binary = matrix_.copy()

        bits: list[int] = Bitmap.split_single_bit(matrix_work[addr])
        for bit in bits:
            # trial to select one bit in target cell
            matrix_buf: Matrix.Binary
            matrix_buf = self.prune_by_pivot(matrix_work, addr, bit)  # type: ignore
            if matrix_buf is None:
                # revert and got next candidate bit
                self._logger.debug('Revert')
                continue
            else:
                # valid and temporarily commit
                matrix_work = matrix_buf
                self.__log_matrix(matrix_work)

            # go down the next node
            matrix_work = self.brute_force(matrix_work, cell_no)
            if self.__done(matrix_work):
                self._logger.debug('successfully done')
                return matrix_work
            else:
                self._logger.debug('not done. Revert!!')
                matrix_work = matrix_.copy()

        self._logger.debug('end of loop of bits')

        return matrix_work


# Sudoku Class - Main Class


class RunMode(Enum):
    PY_BRUTE_FORCE = 1
    C_BRUTE_FORCE = 2
    NO_BRUTE_FORCE = 0


run_mode: RunMode = RunMode.PY_BRUTE_FORCE


class Sudoku(PruneBits, IsolatedBit, BruteForce, Verify, LogCtrl):
    """Sudoku Main Class"""

    problem: Matrix.Decimal = np.full(Matrix.shape, '')
    result: Matrix.Decimal = np.full(Matrix.shape, '')

    _problem: Matrix.Binary
    _working: Matrix.Binary
    _trial: int
    _bf_cnt: int
    _logger: logging.Logger  # type: ignore
    _log_filepath: Log.FilePath = 'a.log'
    _handler_name: Log.HandlerName

    _popcount_sum: dict[object, list[int]]

    def run_mode(self, run_mode_: int = 1) -> None:
        # set run mode of brute force
        global run_mode
        if run_mode_ == 1:
            run_mode = RunMode.PY_BRUTE_FORCE
        elif run_mode_ == 2:
            run_mode = RunMode.C_BRUTE_FORCE
        else:
            run_mode = RunMode.NO_BRUTE_FORCE

    def __zero_sum(self) -> None:
        self._popcount_sum = {
            Row: [0] * Matrix.size,
            Column: [0] * Matrix.size,
            Square: [0] * Matrix.size,
        }

    # solve sudoku (Python version)
    def __py_solve(self) -> None:
        while True:
            # run prune bits algorism
            self.into_msg_area('PRUNE_BITS')
            msg = 'execute Prune Bits'
            self._logger.info(msg)
            print(msg)

            self._bf_cnt = 0
            finished: bool
            changed: bool
            while True:
                self._trial += 1
                finished, changed = self.prune_bits()
                if finished:
                    return
                elif not changed:
                    break

            # run isolated bits algorism
            self.into_msg_area('ISOLATED_BITS')
            msg = 'execute Isolated Bits'
            self._logger.info(msg)
            print(msg)

            finished, changed = self.isolated_bits()
            if finished:
                return
            elif changed:
                continue
            else:
                break

        # run brute force algorism
        global run_mode
        self.into_msg_area('BRUTE_FORCE')
        if run_mode == RunMode.PY_BRUTE_FORCE:
            msg = 'execute Brute Force (Python)'
            self._logger.info(msg)
            print(msg)
            self._working = self.brute_force(self._working)
        else:
            msg = 'skipped Brute Force'
            self._logger.info(msg)
            print(msg)

    # solve sudoku (C version)
    def __c_solve(self) -> None:
        self.into_msg_area('BRUTE_FORCE')
        msg = 'execute Brute Force (C)'
        self._logger.info(msg)
        print(msg)
        self._working = c_brute_force(self._working)

    # solve sudoku
    def __solve(self) -> None:
        """solve the problem"""
        global run_mode
        if run_mode == RunMode.C_BRUTE_FORCE:
            self.__c_solve()
        else:
            self.__py_solve()

        self.into_msg_area()

    def run(self) -> Self:
        """run method to solve"""
        self.__zero_sum()

        Log.setHandler(self._logger, self._handler_name, self._log_filepath)
        start = time.process_time()

        if self._problem is None:
            self._logger.warning('no sudoku data loaded')
        else:
            self.__solve()
            self.result = Bitmap.bmp_to_dec(self._working)  # type: ignore

        end = time.process_time()
        print(f'process time: {end - start:.4f}')
        self._logger.info(f'process time: {end - start:.4f}')
        return self

    def __init__(self) -> None:
        """Sudoku class constructor"""
        Row()
        Column()
        Square()
        Bitmap(9)
        print('sudoku instance created')


# Main for command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser('sudoku')
    parser.add_argument('filename', type=str, help='sudoku problem file')
    run_mode_help = '1:PY_BRUTE_FORCE, 2:C_BRUTE_FORCE, 0:NO_BRUTE_FORCE'
    parser.add_argument('-r', '--run_mode', type=int, default=1, help=run_mode_help)
    parser.add_argument('--log_lv', type=str, default='')
    parser.add_argument('-l', '--log', type=str, default='a.log')
    parser.add_argument('--nolog', action='store_true')
    args = parser.parse_args()

    sudoku = Sudoku()
    sudoku.load(args.filename)
    # sudoku.display_problem_text()

    if args.nolog:
        sudoku.quiet()

    sudoku._log_filepath = args.log
    sudoku.run_mode(args.run_mode)
    sudoku.msg_level('DDD')
    sudoku.run()
    sudoku.display_result_text()
