// Host (macOS/Linux) build of the exact firmware C kernels, for pre-flash verification.
//   host_test L   : stdin = N x 2048 float32 (normalised log-mel) -> stdout = N x d int8
//   host_test A   : stdin = N x 8000 float32 audio -> stdout = N x (2048 float32 x, d int8)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s L|A < input.bin > output.bin\n", argv[0]);
        return 2;
    }
    static float x[N_MELS * N_FRAMES], z[256], a[ENC_BUF_A], b[ENC_BUF_B], pre[PRE_BUF], audio[N_SAMPLES];
    int8_t q[256];
    preproc_init();
    int n = 0;
    if (argv[1][0] == 'L') {
        while (fread(x, sizeof(float), N_MELS * N_FRAMES, stdin) == N_MELS * N_FRAMES) {
            encoder_run(x, z, a, b);
            quantize_int8(z, q, ENC_D);
            fwrite(q, 1, ENC_D, stdout);
            n++;
        }
    } else {
        while (fread(audio, sizeof(float), N_SAMPLES, stdin) == N_SAMPLES) {
            preproc_run(audio, x, pre);
            encoder_run(x, z, a, b);
            quantize_int8(z, q, ENC_D);
            fwrite(x, sizeof(float), N_MELS * N_FRAMES, stdout);
            fwrite(q, 1, ENC_D, stdout);
            n++;
        }
    }
    fprintf(stderr, "host_test %s d=%d model=%s processed=%d\n", argv[1], ENC_D, ENC_NAME, n);
    return 0;
}
