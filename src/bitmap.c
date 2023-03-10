#define __STDC_WANT_LIB_EXT1__ 1
#include "bitmap.h"

#include <string.h>

int split_single_bit(bitmap_t bit, int size, bitmap_t bits[]) {
    int cnt = 0;

    memset_s(bits, size * sizeof(bitmap_t), 0, size * sizeof(bitmap_t));

    for (int i = 0; i < size; i++) {
        bitmap_t target_bit = 1 << i;
        if (bit & target_bit) {
            bits[cnt] = target_bit;
            cnt++;
        }
    }
    return cnt;
}

int popcount(bitmap_t bit) {
    int cnt = 0;

    for (int i = 0; i < BITMAP_DIGIT; i++) {
        bitmap_t target_bit = 1 << i;
        if (bit & target_bit) {
            cnt++;
        }
    }
    return cnt;
}

char *to_binary(bitmap_t bmp, char bin_str[]) {
    char *ptr;

    ptr = bin_str;
    for (int i = 0; i < BITMAP_DIGIT; i++) {
        if (bmp & 0b100000000 >> i) {
            *ptr = '1';
        } else {
            *ptr = '0';
        }
        ptr++;
    }
    bin_str[BITMAP_DIGIT] = '\0';
    return bin_str;
}
