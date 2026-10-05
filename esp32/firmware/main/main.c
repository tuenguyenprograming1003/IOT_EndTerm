// C3 node firmware for ESP32-S3: [preprocess] -> encoder -> INT8 latent -> packet -> UART.
//
// Boot sequence: [MODEL] -> [MEM] -> [CHECK] self-check on embedded vectors -> [BENCH] warm-up +
// measured runs -> [PACKET] -> [READY]. Then serves host commands on the console UART:
//   'L' + 8192 bytes (2048 float32 LE, normalised log-mel)  -> encoder only (Level 1)
//   'A' + 16000 bytes (8000 int16 LE PCM, already fixed to 1 s) -> preprocess + encoder (Level 2)
//   'B' -> rerun benchmark,  'C' -> rerun self-check,  'I' -> print model/memory info
// Each request is answered with one line: [PKT] seq=.. pkt=<hex> pre_us=.. enc_us=.. q_us=.. pkt_us=..
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "sdkconfig.h"
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#else
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#endif
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "model.h"

#define N_WARMUP 50
#define N_RUNS 1000
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
#define CONSOLE_NAME "usb_serial_jtag"
static int console_read(void *dst, int n, TickType_t ticks) { return usb_serial_jtag_read_bytes(dst, n, ticks); }
static void console_init(void) {
    usb_serial_jtag_driver_config_t cfg = {.tx_buffer_size = 4096, .rx_buffer_size = 32768};
    usb_serial_jtag_driver_install(&cfg);
    usb_serial_jtag_vfs_use_driver();
}
#else
#define CONSOLE_NAME "uart"
#define UART_NUM CONFIG_ESP_CONSOLE_UART_NUM
static int console_read(void *dst, int n, TickType_t ticks) { return uart_read_bytes(UART_NUM, dst, n, ticks); }
static void console_init(void) {
    uart_driver_install(UART_NUM, 32768, 0, 0, NULL, 0);
    uart_vfs_dev_use_driver(UART_NUM);
}
#endif

static float *g_x, *g_z, *g_buf_a, *g_buf_b, *g_pre, *g_audio;
static int8_t g_q[256];
static uint8_t g_pkt[256 + PKT_HEADER_BYTES];

static void print_mem(const char *tag) {
    printf("[MEM] stage=%s free_heap=%u min_free_heap=%u internal_free=%u largest_internal_block=%u "
           "psram_total=%u psram_free=%u\n",
           tag, (unsigned)esp_get_free_heap_size(), (unsigned)esp_get_minimum_free_heap_size(),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_total_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

static void print_model(void) {
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    uint32_t flash = 0;
    esp_flash_get_size(NULL, &flash);
    printf("\n[MODEL]\nname=%s\nd=%d\nr=%d\ninput=1x64x32\nlatent=%d\ndtype=INT8 latent, float32 weights/activations\n"
           "encoder_params=%d\nmodel_size_bytes=%d\npreproc_fold=%d\n",
           ENC_NAME, ENC_D, ENC_R, ENC_D, ENC_N_PARAMS, ENC_N_PARAMS * 4, PRE_FOLD);
    printf("[BOARD]\nchip=ESP32-S3 rev=%d cores=%d\nconsole=%s\ncpu_mhz=%d\nflash_bytes=%u\nidf=%s\ncompiler=%s\nopt=%s\n",
           ci.revision, ci.cores, CONSOLE_NAME, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, (unsigned)flash, esp_get_idf_version(),
           __VERSION__,
#if CONFIG_COMPILER_OPTIMIZATION_PERF
           "-O2 (perf)"
#elif CONFIG_COMPILER_OPTIMIZATION_SIZE
           "-Os (size)"
#else
           "other"
#endif
    );
}

static inline int run_node(const float *audio, const float *x_in, int64_t t[4]) {
    // t: pre, enc, quant, packet (microseconds). audio != NULL -> run preprocessing on-node.
    int64_t a = esp_timer_get_time();
    const float *x = x_in;
    if (audio) {
        preproc_run(audio, g_x, g_pre);
        x = g_x;
    }
    int64_t b = esp_timer_get_time();
    encoder_run(x, g_z, g_buf_a, g_buf_b);
    int64_t c = esp_timer_get_time();
    quantize_int8(g_z, g_q, ENC_D);
    int64_t d = esp_timer_get_time();
    int n = build_packet(g_q, ENC_D, g_pkt);
    int64_t e = esp_timer_get_time();
    t[0] = audio ? b - a : 0;
    t[1] = c - b;
    t[2] = d - c;
    t[3] = e - d;
    return n;
}

static void self_check(void) {
    const signed char *exp_q = tv_expected_q();
    int64_t t[4];
    int pass = 0;
    for (int i = 0; i < TV_N; i++) {
        // Level 1: Python log-mel -> on-board encoder
        run_node(NULL, TV_LOGMEL + i * N_MELS * N_FRAMES, t);
        int exact = 0, maxdiff = 0;
        for (int j = 0; j < ENC_D; j++) {
            int dv = abs((int)g_q[j] - (int)exp_q[i * ENC_D + j]);
            exact += dv == 0;
            if (dv > maxdiff) maxdiff = dv;
        }
        // Level 2: raw audio -> on-board preprocessing -> encoder
        run_node(TV_AUDIO + i * N_SAMPLES, NULL, t);
        float max_x_err = 0.f;
        for (int j = 0; j < N_MELS * N_FRAMES; j++) {
            float e = fabsf(g_x[j] - TV_LOGMEL[i * N_MELS * N_FRAMES + j]);
            if (e > max_x_err) max_x_err = e;
        }
        int exact2 = 0, maxdiff2 = 0;
        for (int j = 0; j < ENC_D; j++) {
            int dv = abs((int)g_q[j] - (int)exp_q[i * ENC_D + j]);
            exact2 += dv == 0;
            if (dv > maxdiff2) maxdiff2 = dv;
        }
        int ok = maxdiff <= 1;
        pass += ok;
        vTaskDelay(1);  // feed task watchdog (outside any timed region)
        printf("[CHECK] sample=%03d label=%d L1_exact=%d/%d L1_max_q_diff=%d L2_logmel_max_abs_err=%.6f "
               "L2_exact=%d/%d L2_max_q_diff=%d status=%s\n",
               i, TV_LABEL[i], exact, ENC_D, maxdiff, max_x_err, exact2, ENC_D, maxdiff2, ok ? "PASS" : "FAIL");
    }
    printf("[CHECK] summary pass=%d/%d\n", pass, TV_N);
}

static void benchmark(void) {
    int64_t t[4];
    printf("[BENCH] warmup=%d runs=%d batch=1 input=embedded_vector_0 mode=audio->preproc->encoder->int8->packet\n",
           N_WARMUP, N_RUNS);
    for (int i = 0; i < N_WARMUP; i++) {
        run_node(TV_AUDIO, NULL, t);
        vTaskDelay(1);
    }
    for (int i = 0; i < N_RUNS; i++) {
        run_node(TV_AUDIO, NULL, t);
        printf("[RUN] %d %lld %lld %lld %lld\n", i, t[0], t[1], t[2], t[3]);
        vTaskDelay(1);  // let IDLE run (task watchdog); outside the timed region
    }
    printf("[BENCH] done\n");
    print_mem("after_benchmark");
}

static int read_exact(uint8_t *dst, int n) {
    int got = 0;
    while (got < n) {
        int r = console_read(dst + got, n - got, pdMS_TO_TICKS(5000));
        if (r <= 0) return got;
        got += r;
    }
    return got;
}

static void serve(void) {
    static uint8_t rx[N_SAMPLES * 2 > N_MELS * N_FRAMES * 4 ? N_SAMPLES * 2 : N_MELS * N_FRAMES * 4];
    uint32_t seq = 0;
    for (;;) {
        uint8_t cmd;
        if (console_read(&cmd, 1, portMAX_DELAY) != 1) continue;
        int64_t t[4];
        int n = 0;
        if (cmd == 'L') {
            if (read_exact(rx, N_MELS * N_FRAMES * 4) != N_MELS * N_FRAMES * 4) { printf("[ERR] short L\n"); continue; }
            memcpy(g_x, rx, N_MELS * N_FRAMES * 4);
            n = run_node(NULL, g_x, t);
        } else if (cmd == 'A') {
            if (read_exact(rx, N_SAMPLES * 2) != N_SAMPLES * 2) { printf("[ERR] short A\n"); continue; }
            const int16_t *pcm = (const int16_t *)rx;
            for (int i = 0; i < N_SAMPLES; i++) g_audio[i] = pcm[i] / 32768.f;
            n = run_node(g_audio, NULL, t);
        } else if (cmd == 'B') {
            benchmark();
            printf("[READY]\n");
            continue;
        } else if (cmd == 'C') {
            self_check();
            printf("[READY]\n");
            continue;
        } else if (cmd == 'I') {
            print_model();
            print_mem("info");
            printf("[READY]\n");
            continue;
        } else {
            continue;  // ignore stray bytes (e.g. newlines)
        }
        printf("[PKT] seq=%u mode=%c bytes=%d pkt=", (unsigned)seq++, cmd, n);
        for (int i = 0; i < n; i++) printf("%02x", g_pkt[i]);
        printf(" pre_us=%lld enc_us=%lld q_us=%lld pkt_us=%lld\n", t[0], t[1], t[2], t[3]);
        fflush(stdout);
    }
}

void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(500));
    print_model();
    print_mem("boot");
    g_x = heap_caps_malloc(N_MELS * N_FRAMES * sizeof(float), MALLOC_CAP_INTERNAL);
    g_z = heap_caps_malloc(256 * sizeof(float), MALLOC_CAP_INTERNAL);
    g_buf_a = heap_caps_malloc(ENC_BUF_A * sizeof(float), MALLOC_CAP_INTERNAL);
    g_buf_b = heap_caps_malloc(ENC_BUF_B * sizeof(float), MALLOC_CAP_INTERNAL);
    g_pre = heap_caps_malloc(PRE_BUF * sizeof(float), MALLOC_CAP_INTERNAL);
    g_audio = heap_caps_malloc(N_SAMPLES * sizeof(float), MALLOC_CAP_INTERNAL);
    if (!g_x || !g_z || !g_buf_a || !g_buf_b || !g_pre || !g_audio) {
        printf("[ERR] allocation failed\n");
        return;
    }
    preproc_init();
    printf("[MEM] activation_buffers_bytes=%u\n",
           (unsigned)((N_MELS * N_FRAMES + 256 + ENC_BUF_A + ENC_BUF_B + PRE_BUF + N_SAMPLES) * sizeof(float)));
    print_mem("after_model_init");

    console_init();

    self_check();
    benchmark();
    printf("[PACKET]\npayload_bytes=%d\nheader_bytes=%d\ntotal_bytes=%d\n", ENC_D, PKT_HEADER_BYTES,
           ENC_D + PKT_HEADER_BYTES);
    printf("[READY]\n");
    fflush(stdout);
    serve();
}
