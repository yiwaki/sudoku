#include "bruteforce.h"

#include <stdbool.h>
#include <stdio.h>

bool bruteforce(const matrix_t *x, matrix_t *y, int n) {
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            (*y)[row][col] = (*x)[row][col] * (-n) * 10;
        }
    }
    return true;
}

#ifdef DEBUG
int main(void) {
    matrix_t x;
    matrix_t *y;

    unsigned int c = 0;
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            x[row][col] = c++;
        }
    }

    y = brute_force(&x);
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            printf("%u,", (*y)[row][col]);
        }
        printf("\n");
    }
}
#endif
