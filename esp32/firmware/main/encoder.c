// Plain C float32 encoder: 3x [Conv3x3 s2 p1 + ReLU] -> Conv1x1 + tanh, NCHW layout.
#include <math.h>
#include <string.h>

#include "model.h"

enum { ACT_RELU = 0, ACT_TANH = 1 };

static void conv2d(const float *in, int ci_n, int h, int w, const float *wt, const float *b,
                   int co_n, int k, int s, int p, int act, float *out) {
    const int ho = (h + 2 * p - k) / s + 1, wo = (w + 2 * p - k) / s + 1;
    for (int co = 0; co < co_n; co++) {
        const float *wco = wt + co * ci_n * k * k;
        float *o = out + co * ho * wo;
        for (int oy = 0; oy < ho; oy++) {
            for (int ox = 0; ox < wo; ox++) {
                float acc = b[co];
                const int iy0 = oy * s - p, ix0 = ox * s - p;
                for (int ci = 0; ci < ci_n; ci++) {
                    const float *ich = in + ci * h * w;
                    const float *wk = wco + ci * k * k;
                    for (int ky = 0; ky < k; ky++) {
                        const int iy = iy0 + ky;
                        if (iy < 0 || iy >= h) continue;
                        for (int kx = 0; kx < k; kx++) {
                            const int ix = ix0 + kx;
                            if (ix < 0 || ix >= w) continue;
                            acc += wk[ky * k + kx] * ich[iy * w + ix];
                        }
                    }
                }
                o[oy * wo + ox] = (act == ACT_RELU) ? (acc > 0.f ? acc : 0.f) : tanhf(acc);
            }
        }
    }
}

void encoder_run(const float *x, float *z, float *buf_a, float *buf_b) {
    conv2d(x, 1, 64, 32, ENC_W0, ENC_B0, 32, 3, 2, 1, ACT_RELU, buf_a);      // 32 x 32 x 16
    conv2d(buf_a, 32, 32, 16, ENC_W1, ENC_B1, 64, 3, 2, 1, ACT_RELU, buf_b);  // 64 x 16 x 8
    conv2d(buf_b, 64, 16, 8, ENC_W2, ENC_B2, 128, 3, 2, 1, ACT_RELU, buf_a);  // 128 x 8 x 4
    conv2d(buf_a, 128, 8, 4, ENC_W3, ENC_B3, ENC_R, 1, 1, 0, ACT_TANH, z);    // r x 8 x 4
}

void quantize_int8(const float *z, int8_t *q, int d) {
    for (int i = 0; i < d; i++) {
        float v = nearbyintf(127.f * z[i]);  // round-half-to-even, same as torch.round
        if (v > 127.f) v = 127.f;
        if (v < -127.f) v = -127.f;
        q[i] = (int8_t)v;
    }
}

int build_packet(const int8_t *q, int d, uint8_t *out) {
    out[0] = PKT_VERSION;
    out[1] = PKT_METHOD_TASK_AE;
    out[2] = (uint8_t)(d & 0xff);
    out[3] = (uint8_t)((d >> 8) & 0xff);
    memcpy(out + PKT_HEADER_BYTES, q, d);
    return d + PKT_HEADER_BYTES;
}

const signed char *tv_expected_q(void) {
    switch (ENC_D) {
        case 64: return TV_Q_D64;
        case 128: return TV_Q_D128;
        case 256: return TV_Q_D256;
        default: return 0;
    }
}
