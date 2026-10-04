#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "model_data.h"
#include "audio_frontend.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "WAKE_WORD";

// ==========================================
// Hardware Pin Definitions
// ==========================================
#define I2S_SCK_PIN       GPIO_NUM_26
#define I2S_WS_PIN        GPIO_NUM_25
#define I2S_SD_PIN        GPIO_NUM_33
#define LED_INDICATOR_PIN GPIO_NUM_16

// ==========================================
// Audio & Sliding Window Configuration
// ==========================================
#define SAMPLE_RATE_HZ         16000
#define WINDOW_DURATION_MS     1000
#define WINDOW_TOTAL_SAMPLES   16000
#define STRIDE_DURATION_MS     200
#define STRIDE_SAMPLES         (SAMPLE_RATE_HZ * STRIDE_DURATION_MS / 1000) // 3200 samples
#define NUM_SLICES_PER_STRIDE  10                                           // 10 slices per 200ms

#define DETECTION_THRESHOLD    0.70f  // Trigger probability threshold (updated to 78%)
#define COOLDOWN_PERIOD_MS     1200   // Cooldown after detection to prevent double triggers
#define LED_PULSE_DURATION_MS  800    // How long LED stays ON after detection
#define MIN_SPEECH_RMS         650    // Voice Activity Gate: ignore silence/ambient noise (speech is >1000)

// ==========================================
// TFLite Micro Memory Arena
// ==========================================
constexpr size_t kTensorArenaSize = 85 * 1024;
static uint8_t *s_tensor_arena = nullptr;

// ==========================================
// Global Buffers & Handles
// ==========================================
static i2s_chan_handle_t s_rx_handle = nullptr;
static TimerHandle_t s_led_timer = nullptr;

// Continuous 1-second audio history buffer (16000 samples)
static int16_t s_audio_window[WINDOW_TOTAL_SAMPLES];

// Spectrogram buffer: 49 slices x 40 MFCC features
static int8_t s_mfcc_buffer[MODEL_NUM_SLICES][MODEL_NUM_FEATURES];

// Track detection cooldown
static int64_t s_last_detection_time_ms = 0;

static void led_timer_callback(TimerHandle_t xTimer)
{
    gpio_set_level(LED_INDICATOR_PIN, 0);
}

static esp_err_t init_gpio(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LED_INDICATOR_PIN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    esp_err_t ret = gpio_config(&io_conf);
    if (ret == ESP_OK) {
        gpio_set_level(LED_INDICATOR_PIN, 0);
    }
    return ret;
}

static esp_err_t init_i2s(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 6;
    chan_cfg.dma_frame_num = 512;
    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &s_rx_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create I2S channel: %s", esp_err_to_name(ret));
        return ret;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_SCK_PIN,
            .ws = I2S_WS_PIN,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_SD_PIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ret = i2s_channel_init_std_mode(s_rx_handle, &std_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to init I2S std mode: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = i2s_channel_enable(s_rx_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to enable I2S channel: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "I2S initialized successfully (16kHz, Mono, 32-bit slot for 24-bit INMP441)");
    return ESP_OK;
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "=================================================");
    ESP_LOGI(TAG, "  ESP32 Native Wake Word Detection ('Solomon')   ");
    ESP_LOGI(TAG, "  Continuous 200ms Sliding Window Pipeline       ");
    ESP_LOGI(TAG, "=================================================");

    // 1. Initialize GPIO and indicator timer
    ESP_ERROR_CHECK(init_gpio());
    s_led_timer = xTimerCreate("led_pulse", pdMS_TO_TICKS(LED_PULSE_DURATION_MS), pdFALSE, NULL, led_timer_callback);

    // 2. Initialize Audio Frontend (DSP FFT tables)
    ESP_ERROR_CHECK(audio_frontend_init());

    // 3. Initialize INMP441 I2S Microphone
    ESP_ERROR_CHECK(init_i2s());

    // 4. Allocate TFLM Tensor Arena from Internal Heap
    s_tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    if (!s_tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate %zu bytes for Tensor Arena!", kTensorArenaSize);
        return;
    }
    ESP_LOGI(TAG, "Allocated %zu bytes for Tensor Arena from heap", kTensorArenaSize);

    // 5. Load TFLite Model
    const tflite::Model *model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch! (Expected %d, got %" PRIu32 ")",
                 TFLITE_SCHEMA_VERSION, model->version());
        return;
    }

    // 6. Register Required TFLite Micro Operators
    tflite::MicroMutableOpResolver<20> resolver;
    resolver.AddAdd();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddMul();
    resolver.AddPack();
    resolver.AddReshape();
    resolver.AddShape();
    resolver.AddSoftmax();
    resolver.AddStridedSlice();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddSub();
    resolver.AddPad();
    resolver.AddMean();

    // 7. Instantiate Interpreter
    tflite::MicroInterpreter interpreter(model, resolver, s_tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to allocate model tensors! Please verify tensor arena size and ops.");
        while (1) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }

    TfLiteTensor *input_tensor = interpreter.input(0);
    TfLiteTensor *output_tensor = interpreter.output(0);
    ESP_LOGI(TAG, "Model ready! Arena used: %zu bytes (free heap: %" PRIu32 " bytes)",
             interpreter.arena_used_bytes(), esp_get_free_heap_size());
    ESP_LOGI(TAG, "Input tensor shape: [%d, %d, %d, %d], type: %d",
             input_tensor->dims->data[0], input_tensor->dims->data[1],
             input_tensor->dims->data[2], input_tensor->dims->data[3],
             input_tensor->type);
    ESP_LOGI(TAG, "Output tensor shape: [%d, %d], type: %d",
             output_tensor->dims->data[0], output_tensor->dims->data[1],
             output_tensor->type);

    // Clear audio and spectrogram buffers
    memset(s_audio_window, 0, sizeof(s_audio_window));
    memset(s_mfcc_buffer, MODEL_INPUT_ZERO_POINT, sizeof(s_mfcc_buffer));

    // Buffers to read 200 ms of audio (3200 samples)
    int32_t *dma_buffer = (int32_t *)malloc(STRIDE_SAMPLES * sizeof(int32_t));
    int16_t *stride_audio = (int16_t *)malloc(STRIDE_SAMPLES * sizeof(int16_t));
    if (!dma_buffer || !stride_audio) {
        ESP_LOGE(TAG, "Failed to allocate audio buffers!");
        return;
    }

    ESP_LOGI(TAG, "Listening for wake word 'Solomon'...");

    int speech_hold_frames = 0;       // Number of 200ms strides to keep analyzing speech in the sliding window
    int active_speech_strides = 0;    // Number of consecutive strides with speech energy
    int step_counter = 0;
    static int s_consecutive_hits = 0;

    while (1) {
        // Read 200 ms (3,200 samples) from I2S microphone in 32-bit slot format
        size_t bytes_read = 0;
        esp_err_t ret = i2s_channel_read(s_rx_handle, dma_buffer, STRIDE_SAMPLES * sizeof(int32_t),
                                         &bytes_read, portMAX_DELAY);
        if (ret != ESP_OK || bytes_read != STRIDE_SAMPLES * sizeof(int32_t)) {
            ESP_LOGW(TAG, "I2S read error: %s (read %zu bytes)", esp_err_to_name(ret), bytes_read);
            continue;
        }

        // 1. Convert 24-bit I2S data to signed 16-bit PCM (matching inmp441_audio_recorder.c)
        int64_t chunk_sum = 0;
        for (int i = 0; i < STRIDE_SAMPLES; i++) {
            int32_t s = dma_buffer[i] >> 14;
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            stride_audio[i] = (int16_t)s;
            chunk_sum += s;
        }

        // 2. Subtract DC offset & calculate RMS
        int32_t chunk_dc = (int32_t)(chunk_sum / STRIDE_SAMPLES);
        int64_t sum_sq = 0;
        for (int i = 0; i < STRIDE_SAMPLES; i++) {
            int32_t s = (int32_t)stride_audio[i] - chunk_dc;
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            stride_audio[i] = (int16_t)s;
            sum_sq += s * s;
        }
        int32_t stride_rms = (int32_t)sqrtf((float)(sum_sq / STRIDE_SAMPLES));

        // 2. Voice Activity Gate (VAD) & Active Speech Duration Tracking:
        bool just_started_speech = (speech_hold_frames == 0 && stride_rms >= MIN_SPEECH_RMS);
        if (stride_rms >= MIN_SPEECH_RMS) {
            speech_hold_frames = 5; // Hold active window for 1.0s (5 strides)
            active_speech_strides++;
        } else {
            active_speech_strides = 0;
            if (speech_hold_frames > 0) {
                speech_hold_frames--;
            }
        }

        // 3. Shift audio window left by STRIDE_SAMPLES (keep last 800ms, append new 200ms)
        memmove(s_audio_window, s_audio_window + STRIDE_SAMPLES, (WINDOW_TOTAL_SAMPLES - STRIDE_SAMPLES) * sizeof(int16_t));
        memcpy(s_audio_window + (WINDOW_TOTAL_SAMPLES - STRIDE_SAMPLES), stride_audio, STRIDE_SAMPLES * sizeof(int16_t));

        // 4. If in silence, bypass DSP & TFLM inference completely!
        if (speech_hold_frames == 0) {
            s_consecutive_hits = 0;
            if (++step_counter % 10 == 0) {
                // Heartbeat log every 2 seconds
                ESP_LOGI(TAG, "Listening (Silence)... | Mic RMS: %5" PRId32 " (Speech Gate: %d) | Inference: IDLE",
                         stride_rms, MIN_SPEECH_RMS);
            }
            continue;
        }

        // 5. Calculate audio RMS of the full 1-second audio window (subsampled by 4 for speed)
        int64_t win_sum_sq = 0;
        for (int i = 0; i < WINDOW_TOTAL_SAMPLES; i += 4) {
            int32_t ws = s_audio_window[i];
            win_sum_sq += ws * ws;
        }
        int32_t window_rms = (int32_t)sqrtf((float)(win_sum_sq / (WINDOW_TOTAL_SAMPLES / 4)));

        // 6. Speech is active! Run sliding-window MFCC and model inference
        int64_t start_time = esp_timer_get_time();

        if (just_started_speech) {
            // Speech just started from silence: compute all 49 slices freshly from the full audio window
            for (int slice_idx = 0; slice_idx < MODEL_NUM_SLICES; slice_idx++) {
                int audio_offset = slice_idx * AUDIO_STRIDE_SAMPLES;
                audio_frontend_process_frame(&s_audio_window[audio_offset], &s_mfcc_buffer[slice_idx][0]);
            }
        } else {
            // Continuous sliding: shift existing 39 slices left and compute only 10 new slices
            memmove(&s_mfcc_buffer[0][0], &s_mfcc_buffer[NUM_SLICES_PER_STRIDE][0],
                    (MODEL_NUM_SLICES - NUM_SLICES_PER_STRIDE) * MODEL_NUM_FEATURES * sizeof(int8_t));

            for (int slice_idx = MODEL_NUM_SLICES - NUM_SLICES_PER_STRIDE; slice_idx < MODEL_NUM_SLICES; slice_idx++) {
                int audio_offset = slice_idx * AUDIO_STRIDE_SAMPLES;
                audio_frontend_process_frame(&s_audio_window[audio_offset], &s_mfcc_buffer[slice_idx][0]);
            }
        }

        // Copy spectrogram into TFLM input tensor
        memcpy(input_tensor->data.int8, s_mfcc_buffer, MODEL_NUM_SLICES * MODEL_NUM_FEATURES * sizeof(int8_t));

        // Run TFLM Inference
        TfLiteStatus invoke_status = interpreter.Invoke();
        if (invoke_status != kTfLiteOk) {
            ESP_LOGE(TAG, "Inference invocation failed!");
            continue;
        }

        int64_t end_time = esp_timer_get_time();
        int32_t latency_ms = (int32_t)((end_time - start_time) / 1000);

        // Read Output Probabilities
        int8_t raw_negative = output_tensor->data.int8[0];
        int8_t raw_solomon  = output_tensor->data.int8[1];

        float prob_neg = ((float)raw_negative - (float)MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE;
        float prob_ww  = ((float)raw_solomon  - (float)MODEL_OUTPUT_ZERO_POINT) * MODEL_OUTPUT_SCALE;
        (void)prob_neg;

        int64_t now_ms = esp_timer_get_time() / 1000;
        bool in_cooldown = (now_ms - s_last_detection_time_ms) < COOLDOWN_PERIOD_MS;

        // Anti-Glitch & Syllable Duration Filter:
        // 1. Spikes / clicks last < 50ms (only 1 stride).
        // 2. Short 'saa' noise / sharp breath lasts 1-2 strides (< 400ms).
        // 3. Spoken 'Solomon' has 3 syllables and requires at least 3 active speech strides (>= 600ms)
        //    AND the full 1-second audio window must have speech energy (window_rms >= 450).
        // 4. Strict Debounce: Requires 2 consecutive hits at >= DETECTION_THRESHOLD (no single-frame bypass).
        bool speech_duration_valid = (active_speech_strides >= 3 || speech_hold_frames > 2);
        bool window_energy_valid   = (window_rms >= 450);

        // [TEMPORARILY COMMENTED: 2-hit debounce logic]
        // if (prob_ww >= DETECTION_THRESHOLD && speech_duration_valid && window_energy_valid) {
        //     s_consecutive_hits++;
        // } else {
        //     s_consecutive_hits = 0;
        // }
        // bool is_wake_word = (s_consecutive_hits >= 2);

        // Single-hit trigger without 2-hit debounce:
        bool is_wake_word = (prob_ww >= DETECTION_THRESHOLD && speech_duration_valid && window_energy_valid);

        // Check Detection Trigger
        if (is_wake_word && !in_cooldown) {
            s_last_detection_time_ms = now_ms;
            s_consecutive_hits = 0;
            gpio_set_level(LED_INDICATOR_PIN, 1);
            xTimerReset(s_led_timer, 0);

            ESP_LOGI(TAG, "****************************************************************");
            ESP_LOGI(TAG, ">>> [WAKE WORD DETECTED!] 'SOLOMON' (Score: %5.1f%%, Stride RMS: %" PRId32 ", Win RMS: %" PRId32 ", Latency: %" PRId32 " ms) <<<",
                     prob_ww * 100.0f, stride_rms, window_rms, latency_ms);
            ESP_LOGI(TAG, "****************************************************************");
        } else {
            ESP_LOGI(TAG, "Listening (Voice)...   | Stride RMS: %5" PRId32 " | Win RMS: %5" PRId32 " | Hits: %d | WW Conf: %5.1f%% | Latency: %" PRId32 " ms",
                     stride_rms, window_rms, s_consecutive_hits, prob_ww * 100.0f, latency_ms);
        }
    }
}
