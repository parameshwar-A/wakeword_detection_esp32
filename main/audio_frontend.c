#include "audio_frontend.h"
#include "mel_tables.h"
#include "model_data.h"
#include "esp_dsp.h"
#include "esp_log.h"
#include <math.h>
#include <string.h>

static const char *TAG = "AUDIO_FRONTEND";

// FFT buffer: 512 complex floats (1024 floats: real, imag interleaved)
static float s_fft_buffer[AUDIO_FFT_LEN * 2] __attribute__((aligned(16)));
static float s_magnitude[AUDIO_FFT_LEN / 2 + 1]; // 257 magnitude bins
static float s_log_mel[AUDIO_NUM_MEL_BINS];      // 40 mel energies

esp_err_t audio_frontend_init(void)
{
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, AUDIO_FFT_LEN);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize FFT table (%s)", esp_err_to_name(ret));
        return ret;
    }
    ESP_LOGI(TAG, "Audio frontend initialized (512-pt FFT, 40 Mel bins)");
    return ESP_OK;
}

void audio_frontend_process_frame(const int16_t *pcm_480, int8_t *out_mfcc_40)
{
    // 1. Windowing & Zero-Padding
    // Normalize 16-bit PCM to [-1.0, 1.0], apply Hann window
    for (int i = 0; i < AUDIO_FRAME_LEN; i++) {
        float sample = ((float)pcm_480[i]) / 32768.0f;
        s_fft_buffer[2 * i]     = sample * HANN_WINDOW[i]; // Real part
        s_fft_buffer[2 * i + 1] = 0.0f;                    // Imag part
    }
    // Zero-pad from 480 to 512
    for (int i = AUDIO_FRAME_LEN; i < AUDIO_FFT_LEN; i++) {
        s_fft_buffer[2 * i]     = 0.0f;
        s_fft_buffer[2 * i + 1] = 0.0f;
    }

    // 2. 512-point Complex FFT
    dsps_fft2r_fc32(s_fft_buffer, AUDIO_FFT_LEN);
    dsps_bit_rev_fc32(s_fft_buffer, AUDIO_FFT_LEN);

    // 3. Magnitude Spectrum (Bins 0 to 256)
    for (int k = 0; k <= AUDIO_FFT_LEN / 2; k++) {
        float re = s_fft_buffer[2 * k];
        float im = s_fft_buffer[2 * k + 1];
        s_magnitude[k] = sqrtf(re * re + im * im);
    }

    // 4. Mel Filterbank Accumulation & Log
    for (int m = 0; m < AUDIO_NUM_MEL_BINS; m++) {
        const mel_filter_t *filter = &MEL_FILTERS[m];
        float energy = 0.0f;
        const float *weights = filter->weights;
        const uint16_t start = filter->start_bin;
        const uint16_t count = filter->num_bins;

        for (uint16_t j = 0; j < count; j++) {
            energy += s_magnitude[start + j] * weights[j];
        }
        // Log energy with 1e-6 epsilon
        s_log_mel[m] = logf(energy + 1e-6f);
    }

    // 5. DCT-II Transform & INT8 Quantization
    for (int r = 0; r < AUDIO_NUM_MFCC; r++) {
        float mfcc_val = 0.0f;
        const float *dct_row = DCT_MATRIX[r];

        for (int m = 0; m < AUDIO_NUM_MEL_BINS; m++) {
            mfcc_val += dct_row[m] * s_log_mel[m];
        }

        // Quantize: int8 = clip(round(mfcc / scale + zero_point), -128, 127)
        float q_float = roundf(mfcc_val / MODEL_INPUT_SCALE + (float)MODEL_INPUT_ZERO_POINT);
        int q = (int)q_float;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out_mfcc_40[r] = (int8_t)q;
    }
}
