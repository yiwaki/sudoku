#include "bruteforce.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bitmap.h"
#include "matrix.h"

// bool _valid(const matrix_t x, int cell_no) {
//     address_t addr;
//     cell_no_to_addr(cell_no, &addr);

//     for (int block_type = 0; block_type < BLOCK_TYPE_CNT; block_type++) {
//         int block_no = addr_to_block_no(block_type, &addr);
//         int row_range[2], col_range[2];

//         block_range(block_type, block_no, row_range, col_range);

//         address_t addr;
//         for (addr.row = row_range[0]; addr.row < row_range[1]; addr.row++) {
//             bitmap_t bmp = 0;
//             for (addr.col = col_range[0]; addr.col < col_range[1]; addr.col++) {
//                 bmp |= x[addr.row][addr.col];
//                 if (popcount(x[addr.row][addr.col]) > 1) {
//                     return false;
//                 }
//             }
//             if (bmp != FULL_BIT) {
//                 return false;
//             }
//         }
//     }
//     return true;
// }

bool _done(const matrix_t x) {
    for (int block_type = 0; block_type < BLOCK_TYPE_CNT; block_type++) {
        for (int block_no = 0; block_no < MATRIX_SIZE; block_no++) {
            bitmap_t bmp = 0;

            int row_range[2], col_range[2];
            block_range(block_type, block_no, row_range, col_range);

            for (int row_no = row_range[0]; row_no < row_range[1]; row_no++) {
                for (int col_no = col_range[0]; col_no < col_range[1]; col_no++) {
                    bmp |= x[row_no][col_no];
                    if (popcount(x[row_no][col_no]) > 1) {
                        return false;
                    }
                }
            }
            if (bmp != FULL_BIT) {
                return false;
            }
        }
    }
    return true;
}

bool _prune_by_pivot(const matrix_t *x, const address_t *pivot, bitmap_t bit, matrix_t *y) {
    memcpy(y, x, sizeof(matrix_t));

    bitmap_t pivot_bit = (*x)[pivot->row][pivot->col];
    for (int block_type = 0; block_type < BLOCK_TYPE_CNT; block_type++) {
        int block_no = addr_to_block_no(block_type, pivot);
        int row_range[2], col_range[2];
        block_range(block_type, block_no, row_range, col_range);
        for (int row_no = row_range[0]; row_no < row_range[1]; row_no++) {
            for (int col_no = col_range[0]; col_no < col_range[1]; col_no++) {
                if (row_no == pivot->row && col_no == pivot->col)
                    continue;

                (*y)[row_no][col_no] &= (~pivot_bit);
                if ((*y)[row_no][col_no] == 0)
                    return false;
            }
        }
    }
    return true;
}

void bruteforce(const matrix_t *x, int cell_no, matrix_t *y) {
    cell_no++;
    if (cell_no > MATRIX_SIZE * MATRIX_SIZE - 1) {
        return;
    }

    address_t addr;
    addr_to_block_no(ROW, &addr);

    bitmap_t bits[MATRIX_SIZE];
    int bit_cnt = split_single_bit((*x)[addr.row][addr.col], bits);
    for (int i = 0; i < bit_cnt; i++) {
        bool stat = _prune_by_pivot(x, &addr, bits[i], y);
        if (stat) {
            matrix_t work;
            bruteforce(y, cell_no, &work);

            if (_done(work)) {
                memcpy(y, work, sizeof(matrix_t));
                return;
            }
        }
    }
}

#ifdef DEBUG
int main(void) {
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

    unsigned int c = 0;
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            x[row][col] = c++;
        }
    }

    bitmap_t bits[MATRIX_SIZE];
    int cnt = split_single_bit(0b101101, bits);

    bruteforce(&x, -1, &y);
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            printf("%03o,", y[row][col]);
        }
        printf("\n");
    }
}
#endif
