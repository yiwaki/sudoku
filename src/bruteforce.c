#include "bruteforce.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitmap.h"
#include "matrix.h"

bool _done(matrix_t *const x) {
    for (int block_type = 0; block_type < BLOCK_TYPE_CNT; block_type++) {
        for (int block_no = 0; block_no < MATRIX_SIZE; block_no++) {
            bitmap_t bmp = 0;

            int row_range[2], col_range[2];
            block_range(block_type, block_no, row_range, col_range);

            for (int row_no = row_range[0]; row_no < row_range[1]; row_no++) {
                for (int col_no = col_range[0]; col_no < col_range[1]; col_no++) {
                    bmp |= (*x)[row_no][col_no];

                    if (popcount((*x)[row_no][col_no]) > 1) return false;
                }
            }

            if (bmp != FULL_BIT) return false;
        }
    }
    return true;
}

bool _prune_by_pivot(matrix_t *const x, const address_t *const pivot,
    const bitmap_t bit, matrix_t *const y) {

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

                if ((*y)[row_no][col_no] == 0)
                    return false;
            }
        }
    }
    return true;
}

void bruteforce(matrix_t *const x, int cell_no, matrix_t *const y) {
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

        if (_done(&work)) {
            memcpy(y, work, sizeof(matrix_t));
            return;
        }
    }
}

#ifdef DEBUG
int main(void) {
    matrix_t x = {
        {511, 511,   8, 511,  16, 511, 511, 511,   1},
        {511, 511,  32, 511, 511, 511, 511,   4, 511},
        { 16,   4, 511,  64, 511, 511, 511, 511, 128},
        {  1,   2, 511, 511,  32, 511, 511, 128, 511},
        {511, 511,   4, 511, 511, 511, 511, 511, 511},
        {511, 511, 511, 511, 511, 256, 511, 511,  64},
        {  8, 511, 511, 511, 511, 511, 511, 511, 511},
        {128,  16, 511, 511,   1, 511, 511,   2, 511},
        {511, 511, 511,  32, 511, 511,   1, 511, 511}
    };
    matrix_t y;

    bruteforce(&x, -1, &y);

    char bin_str[BITMAP_DIGIT];
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            printf("%s ", to_binary(y[row][col], bin_str));
        }
        printf("\n");
    }
}
#endif
