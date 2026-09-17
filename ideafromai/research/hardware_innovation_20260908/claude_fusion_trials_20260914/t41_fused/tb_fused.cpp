// T41 TB：t41_fused_gate 的 RTL 回放。自有代码。
// 逐组驱动 in_yv(10×24b signed) / in_thr_row(10×64b，低 48b signed)，
// 比较 out_dec 与激励文件里的期望判决。
// 用法：sim +stim=t41_fused/stim_fused.txt +out=t41_fused/rtl_dec.txt
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include "Vt41_fused_gate.h"
#include "verilated.h"

static Vt41_fused_gate *dut;

static void posedge() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

// Verilator 4.028 把 >64b 的端口展开成裸 WData 数组，需按位落位。
static void zero_wide(unsigned int *w, int nwords) {
    for (int i = 0; i < nwords; i++) w[i] = 0;
}

static void set_bits(unsigned int *w, uint64_t val, int bitoff, int nbits) {
    for (int b = 0; b < nbits; b++) {
        if (!((val >> b) & 1ULL)) continue;
        int pos = bitoff + b;
        w[pos >> 5] |= (1u << (pos & 31));
    }
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vt41_fused_gate;

    const char *stim = nullptr, *outf = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "+stim=", 6)) stim = argv[i] + 6;
        else if (!strncmp(argv[i], "+out=", 5)) outf = argv[i] + 5;
    }
    if (!stim || !outf) { fprintf(stderr, "need +stim= +out=\n"); return 1; }

    FILE *f = fopen(stim, "r");
    if (!f) { fprintf(stderr, "open %s failed\n", stim); return 1; }
    FILE *fo = fopen(outf, "w");
    if (!fo) { fprintf(stderr, "open %s failed\n", outf); return 1; }

    dut->rst_n = 0; dut->in_valid = 0;
    zero_wide(dut->in_yv, 8); zero_wide(dut->in_thr_row, 20);
    posedge(); posedge();
    dut->rst_n = 1;

    char line[4096];
    long groups = 0, mism = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] != 'G') continue;
        long long y[10], t[10];
        unsigned exp;
        const char *p = line + 2;
        char *end;
        for (int i = 0; i < 10; i++) { y[i] = strtoll(p, &end, 10); p = end; }
        for (int i = 0; i < 10; i++) { t[i] = strtoll(p, &end, 10); p = end; }
        exp = (unsigned)strtoul(p, &end, 16);

        zero_wide(dut->in_yv, 8); zero_wide(dut->in_thr_row, 20);
        for (int i = 0; i < 10; i++) {
            set_bits(dut->in_yv, (uint64_t)(y[i] & 0xFFFFFFLL), 24 * i, 24);
            set_bits(dut->in_thr_row, (uint64_t)(t[i] & 0xFFFFFFFFFFFFLL), 64 * i, 48);
        }
        dut->in_valid = 1;
        posedge();                       // 本拍采样，输出在下一拍有效
        uint64_t got = (uint64_t)dut->out_dec;
        if (!dut->out_valid) { fprintf(stderr, "out_valid low at group %ld\n", groups); return 1; }
        if ((unsigned)got != exp) {
            if (mism < 5)
                fprintf(stderr, "MISMATCH g=%ld got=%03x exp=%03x\n", groups,
                        (unsigned)got, exp);
            mism++;
        }
        fprintf(fo, "%03x\n", (unsigned)got);
        groups++;
    }
    fclose(f); fclose(fo);
    printf("groups %ld  mismatch %ld  -> %s\n", groups, mism,
           mism == 0 ? "PASS" : "FAIL");
    delete dut;
    return mism == 0 ? 0 : 1;
}
