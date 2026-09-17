// T24 TB：fx_gate_synth 的 VCD 变体（tb_fx_synth.cpp + --trace）。
// 用法：sim +stim=.. +taudir=.. +out=result.txt +vcd=dump.vcd [+maxg=N]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "Vfx_gate_synth.h"
#include "verilated_vcd_c.h"
#include "verilated.h"

static Vfx_gate_synth *dut;
static VerilatedVcdC *tfp;
static vluint64_t tick_cnt = 0;

static void posedge() {
    dut->clk = 0; dut->eval();
    if (tfp) tfp->dump(tick_cnt * 4);
    dut->clk = 1; dut->eval();
    if (tfp) tfp->dump(tick_cnt * 4 + 2);
    tick_cnt++;
}

struct Group {
    int h, sign;
    std::vector<int> planes;
};

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vfx_gate_synth;

    const char *stim = nullptr, *out = nullptr, *taudir = nullptr, *vcd = nullptr;
    long maxg = -1;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "+stim=", 6)) stim = argv[i] + 6;
        else if (!strncmp(argv[i], "+out=", 5)) out = argv[i] + 5;
        else if (!strncmp(argv[i], "+taudir=", 8)) taudir = argv[i] + 8;
        else if (!strncmp(argv[i], "+vcd=", 5)) vcd = argv[i] + 5;
        else if (!strncmp(argv[i], "+maxg=", 6)) maxg = atol(argv[i] + 6);
    }
    if (!stim || !out || !taudir) {
        fprintf(stderr, "need +stim= +out= +taudir=\n");
        return 1;
    }

    static unsigned long long tau[10][4096];
    int H = -1;
    for (int t = 0; t < 10; t++) {
        char fn[512];
        snprintf(fn, sizeof fn, "%s/tau_t%d.hex", taudir, t);
        FILE *ft = fopen(fn, "r");
        if (!ft) { fprintf(stderr, "open %s failed\n", fn); return 1; }
        char ln[64];
        int n = 0;
        while (fgets(ln, sizeof ln, ft)) {
            if (ln[0] == '\n' || ln[0] == '#') continue;
            tau[t][n++] = strtoull(ln, nullptr, 16);
        }
        fclose(ft);
        if (H < 0) H = n;
        else if (H != n) { fprintf(stderr, "tau size mismatch\n"); return 1; }
    }

    FILE *f = fopen(stim, "r");
    if (!f) { fprintf(stderr, "open %s failed\n", stim); return 1; }
    std::vector<Group> groups;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == 'G') {
            Group g;
            int sign = 0;
            if (sscanf(line, "G %d %x", &g.h, &sign) < 2) { fprintf(stderr, "bad G\n"); return 1; }
            g.sign = sign & 0x3ff;
            groups.push_back(g);
        } else if (line[0] == 'B') {
            unsigned p;
            sscanf(line, "B %x", &p);
            groups.back().planes.push_back(p & 0x3ff);
        }
    }
    fclose(f);
    if (maxg > 0 && (long)groups.size() > maxg) groups.resize(maxg);

    if (vcd) {
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        dut->trace(tfp, 99);
        tfp->open(vcd);
    }

    dut->rst_n = 0; dut->in_sop = 0; dut->in_word_valid = 0;
    dut->in_h = 0; dut->in_word = 0;
    for (int w = 0; w < 20; w++) dut->in_thr_row[w] = 0;
    posedge(); posedge();
    dut->rst_n = 1; posedge();

    FILE *fo = fopen(out, "w");
    fprintf(fo, "# mode=fx_word groups=%zu\n", groups.size());
    vluint64_t total_cycles = 0;
    for (size_t gi = 0; gi < groups.size(); gi++) {
        Group &g = groups[gi];
        while (g.planes.size() < 23) g.planes.push_back(0);
        int yword[10];
        for (int s = 0; s < 10; s++) {
            int mag = 0;
            for (int i = 0; i < 23; i++)
                mag = (mag << 1) | ((g.planes[i] >> s) & 1);
            yword[s] = mag - (((g.sign >> s) & 1) << 23);
        }
        for (int t = 0; t < 10; t++) {
            dut->in_thr_row[2*t]   = (unsigned)(tau[t][g.h] & 0xffffffffull);
            dut->in_thr_row[2*t+1] = (unsigned)(tau[t][g.h] >> 32);
        }
        dut->in_sop = 1; dut->in_word_valid = 0; dut->in_h = g.h;
        posedge();
        for (int s = 0; s < 10; s++) {
            dut->in_sop = 0; dut->in_word_valid = 1;
            dut->in_word = (unsigned)yword[s] & 0xffffff;
            if (s == 9) {
                dut->clk = 0; dut->eval();
                if (!dut->out_valid) { fprintf(stderr, "group %zu out_valid=0\n", gi); return 2; }
            }
            posedge();
        }
        fprintf(fo, "R %03x 10\n", (unsigned)dut->out_dec & 0x3ff);
        total_cycles += 11;
        dut->in_sop = 0; dut->in_word_valid = 0;
    }
    fclose(fo);
    printf("fx_word groups=%zu cycles/group=11\n", groups.size());
    if (tfp) { tfp->close(); delete tfp; }
    delete dut;
    return 0;
}
