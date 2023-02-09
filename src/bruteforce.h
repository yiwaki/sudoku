#ifndef __BRUTEFORCE_H__
#define __BRUTEFORCE_H__

#include "bitmap.h"
#include "matrix.h"

#define MATRIX_SIZE 9

typedef bitmap_t matrix_t[MATRIX_SIZE][MATRIX_SIZE];

void bruteforce(const matrix_t *x, int n, matrix_t *y);

#endif  // __BRUTEFORCE_H__
