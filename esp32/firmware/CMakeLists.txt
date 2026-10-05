# C3 node firmware (ESP-IDF >= 5.3). Select latent size: idf.py -DENC_D=64|128|256 build
cmake_minimum_required(VERSION 3.16)
set(ENC_D 128 CACHE STRING "Latent size d of the exported encoder (64, 128, 256)")
set(PRE_FOLD 1 CACHE STRING "Fold whose Train normalisation constants are embedded")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(c3_node)
