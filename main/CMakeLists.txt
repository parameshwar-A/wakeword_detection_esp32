idf_component_register(
    SRCS "main.cc" "audio_frontend.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_i2s esp_driver_gpio esp_timer
    PRIV_REQUIRES espressif__esp-tflite-micro espressif__esp-dsp
)
