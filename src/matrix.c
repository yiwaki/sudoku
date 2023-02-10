#include "matrix.h"

#include "bruteforce.h"

void cell_no_to_addr(const int cell_no, address_t *addr) {
    addr->row = cell_no / MATRIX_SIZE;
    addr->col = cell_no % MATRIX_SIZE;
}

int addr_to_block_no(const block_t block_type, const address_t *const addr) {
    int block_no;
    switch (block_type) {
        case ROW:
            block_no = addr->row;
            break;

        case COLUMN:
            block_no = addr->col;
            break;

        case SQUARE:
            block_no = addr->row / SQUARE_SIZE * SQUARE_SIZE + addr->col / SQUARE_SIZE;
            break;

        default:
            block_no = -1;
            break;
    }
    return block_no;
}

void block_range(const block_t block_type, const int block_no, int row_range[], int col_range[]) {
    switch (block_type) {
        case ROW:
            row_range[0] = block_no;
            row_range[1] = row_range[0] + 1;
            col_range[0] = 0;
            col_range[1] = MATRIX_SIZE;
            break;

        case COLUMN:
            row_range[0] = 0;
            row_range[1] = MATRIX_SIZE;
            col_range[0] = block_no;
            col_range[1] = col_range[0] + 1;
            break;

        case SQUARE:
            row_range[0] = block_no / SQUARE_SIZE * SQUARE_SIZE;
            row_range[1] = row_range[0] + SQUARE_SIZE;
            col_range[0] = block_no % SQUARE_SIZE * SQUARE_SIZE;
            col_range[1] = col_range[0] + SQUARE_SIZE;
            break;

        default:
            // noop
            break;
    }
}