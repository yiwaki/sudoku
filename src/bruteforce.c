#include "bruteforce.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitmap.h"
#include "matrix.h"

bool _done(const matrix_t x) {
    for (int block_type = 0; block_type < BLOCK_TYPE_CNT; block_type++) {
        for (int block_no = 0; block_no < MATRIX_SIZE; block_no++) {
            bitmap_t bmp = 0;

            int row_range[2], col_range[2];
            block_range(block_type, block_no, row_range, col_range);

            for (int row_no = row_range[0]; row_no < row_range[1]; row_no++) {
                for (int col_no = col_range[0]; col_no < col_range[1]; col_no++) {
                    bmp |= x[row_no][col_no];

                    if (popcount(x[row_no][col_no]) > 1) return false;
                }
            }

            if (bmp != FULL_BIT) return false;
        }
    }
    return true;
}

bool _prune_by_pivot(const matrix_t *x, const address_t *pivot, bitmap_t bit, matrix_t *y) {
    memcpy(y, x, sizeof(matrix_t));

    for (int block_type = 0; block_type < BLOCK_TYPE_CNT; block_type++) {
        int block_no;
        int row_range[2], col_range[2];

        block_no = addr_to_block_no(block_type, pivot);
        block_range(block_type, block_no, row_range, col_range);

        for (int row_no = row_range[0]; row_no < row_range[1]; row_no++) {
            for (int col_no = col_range[0]; col_no < col_range[1]; col_no++) {
                if (row_no == pivot->row && col_no == pivot->col) {
                    (*y)[row_no][col_no] = bit;
                    continue;
                }

                (*y)[row_no][col_no] &= (~bit);

                if ((*y)[row_no][col_no] == 0) {
                    printf("pivot=(%d,%d)\n", pivot->row, pivot->col);
                    printf(
                        "revert: type:%d bk#=%d addr=(%d,%d) bit=%s\n",
                        block_type, block_no, row_no, col_no, to_binary(bit));
                    return false;
                }
            }
        }
    }
    return true;
}

void bruteforce(const matrix_t *x, int cell_no, matrix_t *y) {
    address_t addr;
    bitmap_t bits[MATRIX_SIZE];

    cell_no++;
    if (cell_no > MATRIX_SIZE * MATRIX_SIZE) {
        printf("reached end of cell");
        return;
    }

    cell_no_to_addr(cell_no, &addr);

    int bit_cnt = split_single_bit((*x)[addr.row][addr.col], bits);
    for (int i = 0; i < bit_cnt; i++) {
        matrix_t work;

        if (!_prune_by_pivot(x, &addr, bits[i], y)) continue;

        bruteforce(y, cell_no, &work);

        if (_done(work)) {
            memcpy(y, work, sizeof(matrix_t));
            return;
        }
    }
}

#ifdef DEBUG
int main(void) {
    int r_range[2], c_range[2];
    for (int r = 0; r < MATRIX_SIZE; r++) {
        for (int c = 0; c < MATRIX_SIZE; c++) {
            address_t addr = {r, c};
            int b = addr_to_block_no(SQUARE, &addr);
            printf("%d ", b);
        }
        printf("\n");
    }

    for (int t = 0; t < BLOCK_TYPE_CNT; t++) {
        printf("type:%d\n", t);
        for (int i = 0; i < 9; i++) {
            block_range(t, i, r_range, c_range);
            printf("([%d,%d],[%d,%d]) ", r_range[0], r_range[1], c_range[0], c_range[1]);
            if (i % 3 == 2) printf("\n");
        }
    }

    matrix_t x = {
        {64, 8, 256, 16, 2, 4, 128, 32, 1},
        {16, 1, 128, 256, 32, 8, 64, 4, 2},
        {4, 32, 2, 128, 1, 64, 16, 256, 8},
        {28, 16, 32, 1, 4, 2, 8, 64, 256},
        {2, 4, 1, 8, 64, 256, 32, 16, 128},
        {56, 64, 8, 32, 128, 16, 2, 1, 4},
        {32, 2, 4, 64, 8, 1, 256, 128, 16},
        {1, 128, 16, 2, 256, 32, 4, 8, 64},
        {8, 256, 64, 4, 16, 128, 1, 2, 32}};
    matrix_t y;

    bruteforce(&x, -1, &y);

    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            printf("%s ", to_binary(y[row][col]));
        }
        printf("\n");
    }
}
#endif
