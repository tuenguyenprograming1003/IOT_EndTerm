// On-node preprocessing equivalent to src/features.py:
// centre-padded (constant 0) Hann STFT n_fft=256 hop=256 -> |X|^2 -> slaney mel (64) -> log(P+1e-6)
// -> per-recording CMVN -> per-mel-bin Train normalisation (fold constants).
#include <math.h>

#include "model.h"

static float COS_T[N_FFT], SIN_T[N_FFT];

void preproc_init(void) {
    for (int n = 0; n < N_FFT; n++) {
        COS_T[n] = cosf(2.f * (float)M_PI * n / N_FFT);
        SIN_T[n] = sinf(2.f * (float)M_PI * n / N_FFT);
    }
}

void preproc_run(const float *audio, float *x, float *scratch) {
    float *power = scratch;          // [N_BINS]
    float *frame = scratch + N_BINS; // [N_FFT]
    for (int t = 0; t < N_FRAMES; t++) {
        const int start = t * HOP - N_FFT / 2;
        for (int n = 0; n < N_FFT; n++) {
            const int i = start + n;
            frame[n] = (i >= 0 && i < N_SAMPLES) ? audio[i] * PRE_HANN[n] : 0.f;
        }
        for (int k = 0; k < N_BINS; k++) {  // real DFT via twiddle tables
            float re = 0.f, im = 0.f;
            int idx = 0;
            for (int n = 0; n < N_FFT; n++) {
                re += frame[n] * COS_T[idx];
                im -= frame[n] * SIN_T[idx];
                idx = (idx + k) & (N_FFT - 1);
            }
            power[k] = re * re + im * im;
        }
        for (int m = 0; m < N_MELS; m++) {
            const float *fb = PRE_MELFB + m * N_BINS;
            float acc = 0.f;
            for (int k = 0; k < N_BINS; k++) acc += fb[k] * power[k];
            x[m * N_FRAMES + t] = logf(acc + 1e-6f);
        }
    }
    // per-recording CMVN (population std, +1e-6)
    double sum = 0.0, sq = 0.0;
    const int n_all = N_MELS * N_FRAMES;
    for (int i = 0; i < n_all; i++) sum += x[i];
    const float mean = (float)(sum / n_all);
    for (int i = 0; i < n_all; i++) { const double dv = x[i] - mean; sq += dv * dv; }
    const float sd = (float)sqrt(sq / n_all) + 1e-6f;
    for (int m = 0; m < N_MELS; m++) {
        const float s = PRE_SIGMA[m] > 1e-6f ? PRE_SIGMA[m] : 1e-6f;
        for (int t = 0; t < N_FRAMES; t++) {
            float *v = &x[m * N_FRAMES + t];
            *v = ((*v - mean) / sd - PRE_MU[m]) / s;
        }
    }
}
