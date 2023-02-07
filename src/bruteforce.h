#ifndef __BRUTEFORCE_H__
#define __BRUTEFORCE_H__

#include <stdbool.h>

#include "bitmap.h"

typedef bitmap_t matrix_t[MATRIX_SIZE][MATRIX_SIZE];

void bruteforce(const matrix_t *x, int n, matrix_t *y);

#endif  // __BRUTEFORCE_H__
