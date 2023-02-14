#ifndef BRUTEFORCE_H
#define BRUTEFORCE_H

#include "bitmap.h"
#include "matrix.h"

typedef bitmap_t matrix_t[MATRIX_SIZE][MATRIX_SIZE];

void bruteforce(matrix_t* const x, int n, matrix_t* const y);

#endif  // BRUTEFORCE_H
