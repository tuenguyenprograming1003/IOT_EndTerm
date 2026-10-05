// Node-side C implementation of the C3 codec encoder + preprocessing (ESP32-S3 and host test).
#pragma once
#include <stdint.h>

#define N_MELS 64
#define N_FRAMES 32
#define N_SAMPLES 8000
#define N_FFT 256
#define HOP 256
#define N_BINS (N_FFT / 2 + 1)
#define PKT_HEADER_BYTES 4
#define PKT_VERSION 1
#define PKT_METHOD_TASK_AE 2

// ---- generated weights (esp32/model/model_d{d}.c) ----
extern const float ENC_W0[], ENC_B0[], ENC_W1[], ENC_B1[], ENC_W2[], ENC_B2[], ENC_W3[], ENC_B3[];
extern const int ENC_R, ENC_D, ENC_N_PARAMS;
extern const char ENC_NAME[];
// ---- generated preprocessing constants (esp32/model/preproc_fold{k}.c) ----
extern const float PRE_HANN[], PRE_MELFB[], PRE_MU[], PRE_SIGMA[];
extern const int PRE_FOLD;
// ---- generated test vectors (esp32/model/test_vectors.c) ----
extern const int TV_N;
extern const float TV_AUDIO[], TV_LOGMEL[];
extern const int TV_LABEL[];
extern const signed char TV_Q_D64[], TV_Q_D128[], TV_Q_D256[];

// Scratch sizes (floats)
#define ENC_BUF_A (32 * 32 * 16)  // conv1 output, conv3 output
#define ENC_BUF_B (64 * 16 * 8)   // conv2 output
#define PRE_BUF (N_BINS + N_FFT)

// x: normalised log-mel [64*32] (C-order mel x frame); z: [ENC_D] tanh latent in (r, 8, 4) order
void encoder_run(const float *x, float *z, float *buf_a, float *buf_b);
// q = clip(round(127 z), -127, 127)
void quantize_int8(const float *z, int8_t *q, int d);
// header (version u8, method u8, config u16 LE) + d int8; returns total bytes
int build_packet(const int8_t *q, int d, uint8_t *out);
// audio [8000] float32 -> normalised log-mel x [64*32]: STFT -> mel -> log(P+1e-6) -> CMVN -> Train mu/sigma
void preproc_init(void);
void preproc_run(const float *audio, float *x, float *scratch);
// Returns pointer to the embedded expected INT8 latents for ENC_D (or 0)
const signed char *tv_expected_q(void);
