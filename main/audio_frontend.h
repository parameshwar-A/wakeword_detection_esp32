#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#define AUDIO_FRAME_SAMPLES   480
#define AUDIO_STRIDE_SAMPLES  320
#define MFCC_NUM_FEATURES     40

/**
 * @brief Initialize the audio frontend (FFT tables, etc.)
 */
esp_err_t audio_frontend_init(void);

/**
 * @brief Process one 480-sample frame of audio into 40 quantized INT8 MFCC features.
 * 
 * @param pcm_480 Input pointer to 480 16-bit PCM audio samples (16kHz).
 * @param out_mfcc_40 Output pointer to 40 int8_t quantized MFCC features.
 */
void audio_frontend_process_frame(const int16_t *pcm_480, int8_t *out_mfcc_40);

#ifdef __cplusplus
}
#endif
