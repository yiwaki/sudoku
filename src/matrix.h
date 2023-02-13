#ifndef MATRIX_H
#define MATRIX_H

#define MATRIX_SIZE 9
#define SQUARE_SIZE 3

typedef enum {
    ROW = 0,
    COLUMN,
    SQUARE,
    BLOCK_TYPE_CNT  // count of kinds of block type (ROW/COLUMN/SQUARE)
} block_t;

typedef struct {
    int row;
    int col;
} address_t;

void cell_no_to_addr(const int cell_no, address_t* const addr);
int addr_to_block_no(const block_t block_type, const address_t* const addr);
void block_range(const block_t block_type, const int block_no, int row_range[], int col_range[]);

#endif  // MATRIX_H
