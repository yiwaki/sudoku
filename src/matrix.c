#include "matrix.h"

#include "bruteforce.h"

void cell_no_to_addr(const int cell_no, address_t *addr) {
    addr->row = cell_no / MATRIX_SIZE;
    addr->col = cell_no % MATRIX_SIZE;
}

int addr_to_block_no(const block_t block_type, const address_t *addr) {
    int block_no;
    switch (block_type) {
        case ROW:
            block_no = addr->row;
            break;

        case COLUMN:
            block_no = addr->col;
            break;

        case SQUARE:
            block_no = addr->row / 3 * 3 + addr->col / 3;
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
            row_range[1] = block_no + 1;
            col_range[0] = 0;
            col_range[1] = 9;
            break;

        case COLUMN:
            row_range[0] = 0;
            row_range[1] = 9;
            col_range[0] = block_no;
            col_range[1] = block_no + 1;
            break;

        case SQUARE:
            row_range[0] = block_no / 3;
            row_range[1] = block_no / 3 + 3;
            col_range[0] = block_no % 3;
            col_range[1] = block_no % 3 + 3;
            break;

        default:
            // noop
            break;
    }
}