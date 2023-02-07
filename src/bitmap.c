#include "bitmap.h"

#include <string.h>

int split_single_bit(const bitmap_t bit, bitmap_t *bits) {
    int cnt = 0;

    memset(bits, 0, MATRIX_SIZE * sizeof(bitmap_t));

    for (int i = 0; i < MATRIX_SIZE; i++) {
        bitmap_t target_bit = 1 << i;
        if (bit & target_bit) {
            bits[cnt] = target_bit;
            cnt++;
        }
    }
    return cnt;
}

int popcount(const bitmap_t bit) {
    int cnt = 0;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        bitmap_t target_bit = 1 << i;
        if (bit & target_bit) {
            cnt++;
        }
    }
    return cnt;
}
