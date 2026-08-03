# Core operations: Sequre vs CrypTen vs MPyC

Frameworks present: Sequre (64b) (3 parties), Sequre (128b) (3 parties), Sequre (192b) (3 parties), CrypTen (TFP) (2 parties), CrypTen (TTP) (3 parties), MPyC (3 parties).

The operation set is the intersection of what all three implement in their own library: no cell is a reimplementation. Inputs are `linspace(a, b, n)` over each op's interval; 5 timed repetitions per cell, median reported.

Missing frameworks show `--`. `_skipped_` cells are explained in the table at the end.

### Latency

_milliseconds, median of timed repetitions_

| op | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (TFP) | CrypTen (TTP) | MPyC |
|---|---|---|---|---|---|---|---|
| share | 8 | 0.002 | 0.001 | 0.008 | 0.035 | 0.041 | 0.390 |
| share | 128 | 0.013 | 0.009 | 0.099 | 0.031 | 0.038 | 0.535 |
| share | 1024 | 0.154 | 0.064 | 0.195 | 0.033 | 0.037 | 1.865 |
| share | 8192 | 0.780 | 0.388 | 1.665 | 0.068 | 0.075 | 13.129 |
| add | 8 | -- | -- | -- | 0.120 | 0.119 | 0.242 |
| add | 128 | 0.003 | 0.001 | 0.004 | 0.107 | 0.115 | 0.250 |
| add | 1024 | 0.017 | 0.008 | 0.012 | 0.106 | 0.123 | 0.529 |
| add | 8192 | 0.076 | 0.017 | 0.141 | 0.114 | 0.116 | 2.825 |
| mul_public | 8 | 0.073 | 0.013 | 0.137 | 0.114 | 0.108 | 0.240 |
| mul_public | 128 | 0.132 | 0.085 | 0.144 | 0.116 | 0.109 | 0.231 |
| mul_public | 1024 | 0.305 | 0.363 | 0.718 | 0.112 | 0.109 | 0.529 |
| mul_public | 8192 | 2.104 | 1.911 | 4.454 | 0.114 | 0.114 | 2.922 |
| mul | 8 | 0.092 | 0.023 | 0.177 | 0.482 | 0.924 | 2.183 |
| mul | 128 | 0.096 | 0.044 | 0.194 | 0.481 | 0.789 | 21.831 |
| mul | 1024 | 0.541 | 0.258 | 1.502 | 0.519 | 0.926 | 169.684 |
| mul | 8192 | 4.275 | 3.139 | 7.810 | 0.743 | 1.150 | _skipped_ |
| dot | 8 | 0.143 | 0.021 | 0.126 | 0.473 | 0.901 | 0.819 |
| dot | 128 | 0.106 | 0.045 | 0.226 | 0.481 | 0.946 | 0.801 |
| dot | 1024 | 0.341 | 0.162 | 0.698 | 0.576 | 0.932 | 0.863 |
| dot | 8192 | 2.254 | 2.257 | 3.619 | 0.710 | 1.034 | _skipped_ |
| polynomial | 8 | 0.278 | 0.050 | 0.222 | 1.660 | 2.454 | 4.463 |
| polynomial | 128 | 0.288 | 0.086 | 0.485 | 1.681 | 2.326 | 43.784 |
| polynomial | 1024 | 1.164 | 0.663 | 4.822 | 1.683 | 2.514 | 338.926 |
| polynomial | 8192 | 9.256 | 8.007 | 17.340 | 2.250 | 2.854 | _skipped_ |
| open | 8 | 0.037 | 0.011 | 0.030 | 0.232 | 0.245 | 0.157 |
| open | 128 | 0.056 | 0.020 | 0.110 | 0.223 | 0.205 | 0.143 |
| open | 1024 | 0.170 | 0.072 | 0.343 | 0.254 | 0.262 | 0.418 |
| open | 8192 | 0.931 | 0.468 | 2.618 | 0.306 | 0.247 | 2.477 |

### Throughput

_input elements per second_

| op | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (TFP) | CrypTen (TTP) | MPyC |
|---|---|---|---|---|---|---|---|
| share | 8 | 3,728,270 | 8,388,608 | 1,016,801 | 227,493 | 194,132 | 20,506 |
| share | 128 | 9,761,289 | 14,128,182 | 1,293,665 | 4,134,637 | 3,364,788 | 239,364 |
| share | 1024 | 6,648,556 | 15,966,421 | 5,250,571 | 30,913,271 | 27,706,412 | 548,951 |
| share | 8192 | 10,501,142 | 21,105,490 | 4,919,779 | 121,065,131 | 109,409,075 | 623,970 |
| add | 8 | -- | -- | -- | 66,459 | 67,368 | 33,075 |
| add | 128 | 41,297,762 | 134,217,728 | 33,554,432 | 1,199,997 | 1,111,024 | 512,342 |
| add | 1024 | 59,652,324 | 130,150,524 | 84,215,045 | 9,652,809 | 8,316,757 | 1,934,510 |
| add | 8192 | 107,710,779 | 483,939,977 | 58,040,099 | 71,991,578 | 70,342,928 | 2,900,294 |
| mul_public | 8 | 109,655 | 621,378 | 58,356 | 70,356 | 73,960 | 33,264 |
| mul_public | 128 | 970,833 | 1,503,840 | 890,333 | 1,103,448 | 1,172,075 | 554,213 |
| mul_public | 1024 | 3,358,067 | 2,820,070 | 1,425,952 | 9,166,751 | 9,362,277 | 1,935,271 |
| mul_public | 8192 | 3,893,896 | 4,286,929 | 1,839,288 | 71,911,975 | 71,728,794 | 2,803,360 |
| mul | 8 | 87,154 | 345,922 | 45,222 | 16,606 | 8,663 | 3,665 |
| mul | 128 | 1,332,186 | 2,917,777 | 661,171 | 266,320 | 162,179 | 5,863 |
| mul | 1024 | 1,892,890 | 3,965,805 | 681,741 | 1,972,550 | 1,105,384 | 6,035 |
| mul | 8192 | 1,916,220 | 2,609,534 | 1,048,928 | 11,022,471 | 7,121,416 | _skipped_ |
| dot | 8 | 56,017 | 381,300 | 63,430 | 16,907 | 8,879 | 9,772 |
| dot | 128 | 1,206,451 | 2,840,587 | 566,320 | 265,882 | 135,283 | 159,817 |
| dot | 1024 | 3,003,474 | 6,316,128 | 1,467,362 | 1,778,677 | 1,098,565 | 1,185,986 |
| dot | 8192 | 3,634,797 | 3,629,805 | 2,263,487 | 11,531,939 | 7,921,995 | _skipped_ |
| polynomial | 8 | 28,777 | 160,548 | 36,003 | 4,818 | 3,260 | 1,793 |
| polynomial | 128 | 444,430 | 1,491,308 | 263,948 | 76,130 | 55,020 | 2,923 |
| polynomial | 1024 | 879,756 | 1,544,397 | 212,359 | 608,362 | 407,272 | 3,021 |
| polynomial | 8192 | 885,036 | 1,023,098 | 472,435 | 3,640,350 | 2,870,064 | _skipped_ |
| open | 8 | 217,886 | 729,444 | 264,208 | 34,446 | 32,675 | 50,847 |
| open | 128 | 2,294,320 | 6,468,324 | 1,162,058 | 573,991 | 624,518 | 892,509 |
| open | 1024 | 6,023,797 | 14,221,746 | 2,984,689 | 4,039,447 | 3,911,501 | 2,450,494 |
| open | 8192 | 8,796,656 | 17,512,609 | 3,129,017 | 26,763,982 | 33,149,210 | 3,307,171 |

### Communication

_KiB sent per call by the reporting party_

| op | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (TFP) | CrypTen (TTP) | MPyC |
|---|---|---|---|---|---|---|---|
| share | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9 |
| share | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7.4 |
| share | 1024 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 56.0 |
| share | 8192 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 444.5 |
| add | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 |
| add | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.5 |
| add | 1024 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 18.7 |
| add | 8192 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 148.1 |
| mul_public | 8 | 0.2 | 0.3 | 0.4 | 0.0 | 0.0 | 0.3 |
| mul_public | 128 | 2.0 | 4.0 | 6.0 | 0.0 | 0.0 | 2.5 |
| mul_public | 1024 | 16.0 | 32.0 | 48.0 | 0.0 | 0.0 | 18.7 |
| mul_public | 8192 | 128.0 | 256.0 | 384.0 | 0.0 | 0.0 | 148.2 |
| mul | 8 | 0.2 | 0.4 | 0.6 | 0.2 | 0.2 | 10.8 |
| mul | 128 | 3.0 | 6.0 | 9.0 | 4.0 | 4.0 | 158.2 |
| mul | 1024 | 24.0 | 48.0 | 72.0 | 32.0 | 32.0 | 1,259.0 |
| mul | 8192 | 192.0 | 384.0 | 576.0 | 256.0 | 256.0 | _skipped_ |
| dot | 8 | 0.2 | 0.3 | 0.4 | 0.2 | 0.2 | 1.1 |
| dot | 128 | 2.0 | 4.1 | 6.1 | 4.0 | 4.0 | 1.1 |
| dot | 1024 | 16.0 | 32.1 | 48.1 | 32.0 | 32.0 | 1.1 |
| dot | 8192 | 128.0 | 256.1 | 384.1 | 256.0 | 256.0 | _skipped_ |
| polynomial | 8 | 0.5 | 0.8 | 1.2 | 0.5 | 0.5 | 21.2 |
| polynomial | 128 | 6.1 | 12.1 | 18.1 | 8.0 | 8.0 | 313.9 |
| polynomial | 1024 | 48.1 | 96.1 | 144.1 | 64.0 | 64.0 | 2,499.5 |
| polynomial | 8192 | 384.1 | 768.1 | 1,152.1 | 512.0 | 512.0 | _skipped_ |
| open | 8 | 0.2 | 0.3 | 0.4 | 0.1 | 0.1 | 0.3 |
| open | 128 | 2.0 | 4.0 | 6.0 | 2.0 | 2.0 | 2.5 |
| open | 1024 | 16.0 | 32.0 | 48.0 | 16.0 | 16.0 | 18.6 |
| open | 8192 | 128.0 | 256.0 | 384.0 | 128.0 | 128.0 | 148.2 |

### Accuracy (max absolute error)

_vs float64 computed in the clear_

| op | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (TFP) | CrypTen (TTP) | MPyC |
|---|---|---|---|---|---|---|---|
| share | 8 | 1.3e-05 | 2.0e-10 | 2.0e-10 | 1.3e-05 | 1.3e-05 | 1.0e-10 |
| share | 128 | 1.5e-05 | 2.3e-10 | 2.3e-10 | 1.5e-05 | 1.5e-05 | 1.2e-10 |
| share | 1024 | 1.5e-05 | 2.3e-10 | 2.3e-10 | 1.5e-05 | 1.5e-05 | 1.2e-10 |
| share | 8192 | 1.5e-05 | 2.3e-10 | 2.3e-10 | 1.5e-05 | 1.5e-05 | 1.2e-10 |
| add | 8 | 1.5e-05 | 2.3e-10 | 2.3e-10 | 2.2e-16 | 2.2e-16 | 2.2e-16 |
| add | 128 | 1.5e-05 | 2.3e-10 | 2.3e-10 | 1.1e-16 | 1.1e-16 | 1.1e-16 |
| add | 1024 | 1.5e-05 | 2.3e-10 | 2.3e-10 | 1.1e-16 | 1.1e-16 | 1.1e-16 |
| add | 8192 | 1.5e-05 | 2.3e-10 | 2.3e-10 | 1.1e-16 | 1.1e-16 | 1.1e-16 |
| mul_public | 8 | 3.9e-05 | 6.0e-10 | 6.0e-10 | 3.9e-05 | 3.9e-05 | 3.0e-10 |
| mul_public | 128 | 4.5e-05 | 6.9e-10 | 6.9e-10 | 4.5e-05 | 4.5e-05 | 3.5e-10 |
| mul_public | 1024 | 4.6e-05 | 7.0e-10 | 7.0e-10 | 4.6e-05 | 4.6e-05 | 3.5e-10 |
| mul_public | 8192 | 4.6e-05 | 7.0e-10 | 7.0e-10 | 4.6e-05 | 4.6e-05 | 3.5e-10 |
| mul | 8 | 1.1e-05 | 4.4e-10 | 2.7e-10 | 1.9e-05 | 1.9e-05 | 2.6e-10 |
| mul | 128 | 3.2e-05 | 5.4e-10 | 5.4e-10 | 3.2e-05 | 3.0e-05 | 3.0e-10 |
| mul | 1024 | 3.8e-05 | 6.3e-10 | 5.9e-10 | 3.8e-05 | 3.8e-05 | 3.9e-10 |
| mul | 8192 | 4.3e-05 | 6.3e-10 | 6.4e-10 | 4.2e-05 | 4.2e-05 | _skipped_ |
| dot | 8 | 4.4e-05 | 1.1e-09 | 8.6e-10 | 4.4e-05 | 4.4e-05 | 3.0e-10 |
| dot | 128 | 1.1e-03 | 1.7e-08 | 1.7e-08 | 1.0e-03 | 1.1e-03 | 2.4e-09 |
| dot | 1024 | 8.4e-03 | 1.3e-07 | 1.3e-07 | 8.4e-03 | 8.4e-03 | 4.2e-09 |
| dot | 8192 | 6.5e-02 | 9.7e-07 | 9.7e-07 | 6.5e-02 | 6.5e-02 | _skipped_ |
| polynomial | 8 | 4.7e-04 | 5.9e-09 | 5.9e-09 | 3.6e-04 | 4.7e-04 | 2.3e-09 |
| polynomial | 128 | 8.1e-04 | 1.3e-08 | 1.3e-08 | 7.7e-04 | 7.7e-04 | 7.1e-09 |
| polynomial | 1024 | 9.4e-04 | 1.4e-08 | 1.4e-08 | 1.0e-03 | 9.8e-04 | 6.6e-09 |
| polynomial | 8192 | 9.5e-04 | 1.4e-08 | 1.4e-08 | 1.0e-03 | 1.0e-03 | _skipped_ |
| open | 8 | 1.3e-05 | 2.0e-10 | 2.0e-10 | 1.3e-05 | 1.3e-05 | 1.0e-10 |
| open | 128 | 1.5e-05 | 2.3e-10 | 2.3e-10 | 1.5e-05 | 1.5e-05 | 1.2e-10 |
| open | 1024 | 1.5e-05 | 2.3e-10 | 2.3e-10 | 1.5e-05 | 1.5e-05 | 1.2e-10 |
| open | 8192 | 1.5e-05 | 2.3e-10 | 2.3e-10 | 1.5e-05 | 1.5e-05 | 1.2e-10 |

### Sequre (192b) speedup

_median wall-clock of the baseline divided by Sequre (192b)'s; &gt;1 means Sequre is faster. Note the differing party counts and share widths._

| op | n | vs CrypTen (TFP) | vs CrypTen (TTP) | vs MPyC |
|---|---|---|---|---|
| share | 8 | 4.5x | 5.2x | 49.6x |
| share | 128 | 0.3x | 0.4x | 5.4x |
| share | 1024 | 0.2x | 0.2x | 9.6x |
| share | 8192 | 0.0x | 0.0x | 7.9x |
| add | 8 | -- | -- | -- |
| add | 128 | 28.0x | 30.2x | 65.5x |
| add | 1024 | 8.7x | 10.1x | 43.5x |
| add | 8192 | 0.8x | 0.8x | 20.0x |
| mul_public | 8 | 0.8x | 0.8x | 1.8x |
| mul_public | 128 | 0.8x | 0.8x | 1.6x |
| mul_public | 1024 | 0.2x | 0.2x | 0.7x |
| mul_public | 8192 | 0.0x | 0.0x | 0.7x |
| mul | 8 | 2.7x | 5.2x | 12.3x |
| mul | 128 | 2.5x | 4.1x | 112.8x |
| mul | 1024 | 0.3x | 0.6x | 113.0x |
| mul | 8192 | 0.1x | 0.1x | -- |
| dot | 8 | 3.8x | 7.1x | 6.5x |
| dot | 128 | 2.1x | 4.2x | 3.5x |
| dot | 1024 | 0.8x | 1.3x | 1.2x |
| dot | 8192 | 0.2x | 0.3x | -- |
| polynomial | 8 | 7.5x | 11.0x | 20.1x |
| polynomial | 128 | 3.5x | 4.8x | 90.3x |
| polynomial | 1024 | 0.3x | 0.5x | 70.3x |
| polynomial | 8192 | 0.1x | 0.2x | -- |
| open | 8 | 7.7x | 8.1x | 5.2x |
| open | 128 | 2.0x | 1.9x | 1.3x |
| open | 1024 | 0.7x | 0.8x | 1.2x |
| open | 8192 | 0.1x | 0.1x | 0.9x |

## Online cost (dealer detached)

_Sequre rows have the dealer detached via `mpc.detach_dealer()`, so only the compute parties' work is timed. The other three have no separable offline phase and repeat their end-to-end numbers here._

### Latency, dealer detached

_milliseconds, median of timed repetitions_

| op | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (TFP) | CrypTen (TTP) | MPyC |
|---|---|---|---|---|---|---|---|
| share | 8 | 0.002 | 0.001 | 0.004 | 0.035 | 0.041 | 0.390 |
| share | 128 | 0.017 | 0.005 | 0.012 | 0.031 | 0.038 | 0.535 |
| share | 1024 | 0.075 | 0.028 | 0.090 | 0.033 | 0.037 | 1.865 |
| share | 8192 | 0.495 | 0.453 | 0.655 | 0.068 | 0.075 | 13.129 |
| add | 8 | 0.001 | -- | 0.001 | 0.120 | 0.119 | 0.242 |
| add | 128 | 0.003 | 0.001 | 0.002 | 0.107 | 0.115 | 0.250 |
| add | 1024 | 0.009 | 0.003 | 0.010 | 0.106 | 0.123 | 0.529 |
| add | 8192 | 0.051 | 0.018 | 0.087 | 0.114 | 0.116 | 2.825 |
| mul_public | 8 | 0.030 | 0.013 | 0.034 | 0.114 | 0.108 | 0.240 |
| mul_public | 128 | 0.158 | 0.023 | 0.065 | 0.116 | 0.109 | 0.231 |
| mul_public | 1024 | 0.311 | 0.142 | 0.444 | 0.112 | 0.109 | 0.529 |
| mul_public | 8192 | 1.392 | 0.777 | 3.776 | 0.114 | 0.114 | 2.922 |
| mul | 8 | 0.051 | 0.018 | 0.043 | 0.482 | 0.924 | 2.183 |
| mul | 128 | 0.100 | 0.035 | 0.089 | 0.481 | 0.789 | 21.831 |
| mul | 1024 | 0.373 | 0.191 | 0.763 | 0.519 | 0.926 | 169.684 |
| mul | 8192 | 2.502 | 1.629 | 7.294 | 0.743 | 1.150 | _skipped_ |
| dot | 8 | 0.170 | 0.017 | 0.044 | 0.473 | 0.901 | 0.819 |
| dot | 128 | 0.065 | 0.028 | 0.073 | 0.481 | 0.946 | 0.801 |
| dot | 1024 | 0.245 | 0.113 | 0.429 | 0.576 | 0.932 | 0.863 |
| dot | 8192 | 1.310 | 1.651 | 3.452 | 0.710 | 1.034 | _skipped_ |
| polynomial | 8 | 0.328 | 0.036 | 0.096 | 1.660 | 2.454 | 4.463 |
| polynomial | 128 | 0.192 | 0.077 | 0.239 | 1.681 | 2.326 | 43.784 |
| polynomial | 1024 | 0.778 | 0.434 | 1.147 | 1.683 | 2.514 | 338.926 |
| polynomial | 8192 | 4.873 | 2.848 | 18.005 | 2.250 | 2.854 | _skipped_ |
| open | 8 | 0.021 | 0.010 | 0.027 | 0.232 | 0.245 | 0.157 |
| open | 128 | 0.078 | 0.018 | 0.036 | 0.223 | 0.205 | 0.143 |
| open | 1024 | 0.152 | 0.070 | 0.210 | 0.254 | 0.262 | 0.418 |
| open | 8192 | 0.682 | 0.428 | 1.441 | 0.306 | 0.247 | 2.477 |

### Communication, dealer detached

_KiB sent per call by the reporting party_

| op | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (TFP) | CrypTen (TTP) | MPyC |
|---|---|---|---|---|---|---|---|
| share | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9 |
| share | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7.4 |
| share | 1024 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 56.0 |
| share | 8192 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 444.5 |
| add | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 |
| add | 128 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.5 |
| add | 1024 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 18.7 |
| add | 8192 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 148.1 |
| mul_public | 8 | 0.2 | 0.3 | 0.4 | 0.0 | 0.0 | 0.3 |
| mul_public | 128 | 2.0 | 4.0 | 6.0 | 0.0 | 0.0 | 2.5 |
| mul_public | 1024 | 16.0 | 32.0 | 48.0 | 0.0 | 0.0 | 18.7 |
| mul_public | 8192 | 128.0 | 256.0 | 384.0 | 0.0 | 0.0 | 148.2 |
| mul | 8 | 0.2 | 0.4 | 0.6 | 0.2 | 0.2 | 10.8 |
| mul | 128 | 3.0 | 6.0 | 9.0 | 4.0 | 4.0 | 158.2 |
| mul | 1024 | 24.0 | 48.0 | 72.0 | 32.0 | 32.0 | 1,259.0 |
| mul | 8192 | 192.0 | 384.0 | 576.0 | 256.0 | 256.0 | _skipped_ |
| dot | 8 | 0.2 | 0.3 | 0.4 | 0.2 | 0.2 | 1.1 |
| dot | 128 | 2.0 | 4.1 | 6.1 | 4.0 | 4.0 | 1.1 |
| dot | 1024 | 16.0 | 32.1 | 48.1 | 32.0 | 32.0 | 1.1 |
| dot | 8192 | 128.0 | 256.1 | 384.1 | 256.0 | 256.0 | _skipped_ |
| polynomial | 8 | 0.5 | 0.8 | 1.2 | 0.5 | 0.5 | 21.2 |
| polynomial | 128 | 6.1 | 12.1 | 18.1 | 8.0 | 8.0 | 313.9 |
| polynomial | 1024 | 48.1 | 96.1 | 144.1 | 64.0 | 64.0 | 2,499.5 |
| polynomial | 8192 | 384.1 | 768.1 | 1,152.1 | 512.0 | 512.0 | _skipped_ |
| open | 8 | 0.2 | 0.3 | 0.4 | 0.1 | 0.1 | 0.3 |
| open | 128 | 2.0 | 4.0 | 6.0 | 2.0 | 2.0 | 2.5 |
| open | 1024 | 16.0 | 32.0 | 48.0 | 16.0 | 16.0 | 18.6 |
| open | 8192 | 128.0 | 256.0 | 384.0 | 128.0 | 128.0 | 148.2 |

### Sequre (192b) speedup (dealer detached)

_median wall-clock of the baseline divided by Sequre (192b)'s; &gt;1 means Sequre is faster. Note the differing party counts and share widths._ Sequre is detached; the baselines are not.

| op | n | vs CrypTen (TFP) | vs CrypTen (TTP) | vs MPyC |
|---|---|---|---|---|
| share | 8 | 8.7x | 10.2x | 96.3x |
| share | 128 | 2.6x | 3.2x | 44.9x |
| share | 1024 | 0.4x | 0.4x | 20.8x |
| share | 8192 | 0.1x | 0.1x | 20.0x |
| add | 8 | 126.2x | 124.5x | 253.6x |
| add | 128 | 49.7x | 53.7x | 116.4x |
| add | 1024 | 10.6x | 12.3x | 52.9x |
| add | 8192 | 1.3x | 1.3x | 32.5x |
| mul_public | 8 | 3.3x | 3.2x | 7.1x |
| mul_public | 128 | 1.8x | 1.7x | 3.5x |
| mul_public | 1024 | 0.3x | 0.2x | 1.2x |
| mul_public | 8192 | 0.0x | 0.0x | 0.8x |
| mul | 8 | 11.2x | 21.4x | 50.6x |
| mul | 128 | 5.4x | 8.9x | 244.8x |
| mul | 1024 | 0.7x | 1.2x | 222.5x |
| mul | 8192 | 0.1x | 0.2x | -- |
| dot | 8 | 10.8x | 20.5x | 18.7x |
| dot | 128 | 6.6x | 13.0x | 11.0x |
| dot | 1024 | 1.3x | 2.2x | 2.0x |
| dot | 8192 | 0.2x | 0.3x | -- |
| polynomial | 8 | 17.3x | 25.5x | 46.4x |
| polynomial | 128 | 7.0x | 9.7x | 183.1x |
| polynomial | 1024 | 1.5x | 2.2x | 295.5x |
| polynomial | 8192 | 0.1x | 0.2x | -- |
| open | 8 | 8.6x | 9.1x | 5.8x |
| open | 128 | 6.2x | 5.7x | 4.0x |
| open | 1024 | 1.2x | 1.2x | 2.0x |
| open | 8192 | 0.2x | 0.2x | 1.7x |

### Why cells are skipped

_A cap named in the reason is a runner flag and can be raised; anything else is a property of the framework._

| framework | op | n | reason |
|---|---|---|---|
| MPyC | dot | 8192 | n > --max-n=1024; MPyC is a pure-Python runtime |
| MPyC | mul | 8192 | n > --max-n=1024; MPyC is a pure-Python runtime |
| MPyC | polynomial | 8192 | n > --max-n=1024; MPyC is a pure-Python runtime |

### Run metadata

- `crypten-ttp.jsonl`: poly_coeffs=[1.0, 2.0, 3.0, 4.0], public_scalar=3.0, framework=crypten, parties=2, provider=TTP, protocol=beaver, precision_bits=16, ring_bits=64, note=every op is a CrypTen tensor method; nothing reimplemented
- `crypten.jsonl`: poly_coeffs=[1.0, 2.0, 3.0, 4.0], public_scalar=3.0, framework=crypten, parties=2, provider=TFP, protocol=beaver, precision_bits=16, ring_bits=64, note=every op is a CrypTen tensor method; nothing reimplemented
- `mpyc.jsonl`: poly_coeffs=[1.0, 2.0, 3.0, 4.0], public_scalar=3.0, framework=mpyc, parties=3, threshold=1, sectype=SecFxp(64), note=every op is an MPyC primitive; nothing reimplemented
- `sequre_w128_CP0.jsonl`: public_scalar=3, framework=sequre, parties=3, mpc_int_size=128, nbit_k=64, nbit_f=32, nbit_v=28, modulus=ring_2^127
- `sequre_w128_CP1.jsonl`: public_scalar=3, framework=sequre, parties=3, mpc_int_size=128, nbit_k=64, nbit_f=32, nbit_v=28, modulus=ring_2^127
- `sequre_w128_CP2.jsonl`: public_scalar=3, framework=sequre, parties=3, mpc_int_size=128, nbit_k=64, nbit_f=32, nbit_v=28, modulus=ring_2^127
- `sequre_w192_CP0.jsonl`: public_scalar=3, framework=sequre, parties=3, mpc_int_size=192, nbit_k=64, nbit_f=32, nbit_v=64, modulus=ring_2^191
- `sequre_w192_CP1.jsonl`: public_scalar=3, framework=sequre, parties=3, mpc_int_size=192, nbit_k=64, nbit_f=32, nbit_v=64, modulus=ring_2^191
- `sequre_w192_CP2.jsonl`: public_scalar=3, framework=sequre, parties=3, mpc_int_size=192, nbit_k=64, nbit_f=32, nbit_v=64, modulus=ring_2^191
- `sequre_w64_CP0.jsonl`: public_scalar=3, framework=sequre, parties=3, mpc_int_size=64, nbit_k=32, nbit_f=16, nbit_v=10, modulus=ring_2^63
- `sequre_w64_CP1.jsonl`: public_scalar=3, framework=sequre, parties=3, mpc_int_size=64, nbit_k=32, nbit_f=16, nbit_v=10, modulus=ring_2^63
- `sequre_w64_CP2.jsonl`: public_scalar=3, framework=sequre, parties=3, mpc_int_size=64, nbit_k=32, nbit_f=16, nbit_v=10, modulus=ring_2^63
