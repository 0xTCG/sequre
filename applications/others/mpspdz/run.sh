#!/usr/bin/env bash

Scripts/compile-run.py -R 256 dealer-ring enc
Scripts/compile-run.py -R 256 dealer-ring add_plain
Scripts/compile-run.py -R 256 dealer-ring add
Scripts/compile-run.py -R 256 dealer-ring mul_plain
Scripts/compile-run.py -R 256 dealer-ring mul
Scripts/compile-run.py -R 256 dealer-ring rot
Scripts/compile-run.py -R 256 dealer-ring dec

Scripts/compile-run.py -R 256 dealer-ring l2
Scripts/compile-run.py -R 256 dealer-ring matmul
