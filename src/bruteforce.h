#ifndef BRUTEFORCE_H
#define BRUTEFORCE_H

#include "bitmap.h"
#include "matrix.h"

#define MATRIX_SIZE 9

typedef bitmap_t matrix_t[MATRIX_SIZE][MATRIX_SIZE];

void bruteforce(const matrix_t *x, int n, matrix_t *y);

#endif  // BRUTEFORCE_H
