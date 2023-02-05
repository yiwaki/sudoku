#include "bruteforce.h"

#include <stdio.h>

matrix_t* brute_force(matrix_t* x) {
    static matrix_t y;

    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            y[row][col] = (*x)[row][col] * 10;
        }
    }
    return &y;
}

#ifdef DEBUG
int main(void) {
    matrix_t x;
    matrix_t* y;

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
