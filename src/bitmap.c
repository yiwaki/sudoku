#include "bitmap.h"

#include <string.h>

#define BIT_LEN 9

int split_single_bit(const bitmap_t bit, bitmap_t bits[BIT_LEN]) {
    int cnt = 0;

    memset(bits, 0, BIT_LEN * sizeof(bitmap_t));

    for (int i = 0; i < BIT_LEN; i++) {
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
    for (int i = 0; i < BIT_LEN; i++) {
        bitmap_t target_bit = 1 << i;
        if (bit & target_bit) {
            cnt++;
        }
    }
    return cnt;
}

char *to_binary(const bitmap_t bmp) {
    static char bin[BIT_LEN + 1];
    char *ptr;
    ptr = bin;
    for (int i = 0; i < BIT_LEN; i++) {
        if (bmp & 0b100000000 >> i) {
            *ptr = '1';
        } else {
            *ptr = '0';
        }
        ptr++;
    }
    bin[BIT_LEN] = '\0';
    return bin;
}
