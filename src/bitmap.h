#ifndef __BITMAP_H__
#define __BITMAP_H__

#define FULL_BIT 0b111111111

typedef unsigned short bitmap_t;

int split_single_bit(const bitmap_t bit, bitmap_t *bits);
int popcount(const bitmap_t bit);
char *to_binary(const bitmap_t bmp);

#endif  // __BITMAP_H__
