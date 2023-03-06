#ifndef BITMAP_H
#define BITMAP_H

#define BITMAP_DIGIT (9)
#define FULL_BIT (0b111111111)

typedef unsigned short bitmap_t;

int split_single_bit(bitmap_t bit, int size, bitmap_t bits[]);
int popcount(bitmap_t bit);
char *to_binary(bitmap_t bmp, char bin_str[]);

#endif  // BITMAP_H
