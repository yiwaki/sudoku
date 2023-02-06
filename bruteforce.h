#ifndef __BRUTEFORCE_H__
#define __BRUTEFORCE_H__

#include <stdbool.h>

#define MATRIX_SIZE 9

typedef unsigned short matrix_t[MATRIX_SIZE][MATRIX_SIZE];

bool bruteforce(const matrix_t *x, matrix_t *y, int n);

#endif  // __BRUTEFORCE_H__
