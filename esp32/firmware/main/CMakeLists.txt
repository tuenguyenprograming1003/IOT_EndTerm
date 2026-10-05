set(MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../../model")
idf_component_register(
    SRCS "main.c" "encoder.c" "preproc.c"
         "${MODEL_DIR}/model_d${ENC_D}.c"
         "${MODEL_DIR}/preproc_fold${PRE_FOLD}.c"
         "${MODEL_DIR}/test_vectors.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_driver_uart esp_driver_usb_serial_jtag esp_timer spi_flash)
