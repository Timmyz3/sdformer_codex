// T39 TB：cert_gate_sglr 激励器（SGLR per-lane ready 回放）。自有代码，派生自
// t15_synth/tb_cert_synth.cpp。差异：
//   +gated=1  时，只向 DUT 送 out_need 里的 lane（其余 lane 比特置 0），
//   并逐平面累加 popcount(out_need) = 实际传输比特数。
// 输出两份文件：
//   +out=..     与 T25 参考同格式 "R <dec> <fed>"（gated 与非 gated 须逐字节同）
//   +stats=..   每组一行 "S <fed> <bits_gated> <bits_full>"
// 用法：sim +stim=.. +mode=fx_full|fx_cert|bf_full|bf_cert +taudir=.. +gated=0|1
//          +out=.. +stats=..
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "Vcert_gate_sglr.h"
#include "verilated.h"

static Vcert_gate_sglr *dut;
static vluint64_t tick_cnt = 0;

static void posedge() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    tick_cnt++;
}

struct Group {
    int h, e, sign;
    std::vector<int> planes;
};

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vcert_gate_sglr;

    const char *stim = nullptr, *mode = nullptr, *out = nullptr, *taudir = nullptr;
    const char *stats = nullptr, *gated_s = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "+stim=", 6)) stim = argv[i] + 6;
        else if (!strncmp(argv[i], "+mode=", 6)) mode = argv[i] + 6;
        else if (!strncmp(argv[i], "+out=", 5)) out = argv[i] + 5;
        else if (!strncmp(argv[i], "+taudir=", 8)) taudir = argv[i] + 8;
        else if (!strncmp(argv[i], "+stats=", 7)) stats = argv[i] + 7;
        else if (!strncmp(argv[i], "+gated=", 7)) gated_s = argv[i] + 7;
    }
    if (!stim || !mode || !out || !taudir) {
        fprintf(stderr, "need +stim= +mode= +out= +taudir=\n");
        return 1;
    }
    bool is_bf = !strcmp(mode, "bf_full") || !strcmp(mode, "bf_cert");
    bool is_cert = !strcmp(mode, "fx_cert") || !strcmp(mode, "bf_cert");
    bool gated = gated_s && !strcmp(gated_s, "1");

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
        else if (H != n) { fprintf(stderr, "tau size mismatch t=%d\n", t); return 1; }
    }

    FILE *f = fopen(stim, "r");
    if (!f) { fprintf(stderr, "open %s failed\n", stim); return 1; }
    std::vector<Group> groups;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == 'G') {
            Group g;
            int sign = 0;
            int nf = sscanf(line, "G %d %x %d", &g.h, &sign, &g.e);
            g.sign = sign & 0x3ff;
            if (is_bf) { if (nf < 3) { fprintf(stderr, "bf stim needs e\n"); return 1; } }
            else { g.e = 24; if (nf < 2) { fprintf(stderr, "fx stim needs sign\n"); return 1; } }
            groups.push_back(g);
        } else if (line[0] == 'B') {
            unsigned p;
            sscanf(line, "B %x", &p);
            groups.back().planes.push_back(p & 0x3ff);
        }
    }
    fclose(f);

    dut->rst_n = 0; dut->in_sop = 0; dut->in_plane_valid = 0;
    dut->in_h = 0; dut->in_sign = 0; dut->in_e = 0; dut->in_plane = 0;
    for (int w = 0; w < 20; w++) dut->in_thr_row[w] = 0;
    posedge(); posedge();
    dut->rst_n = 1; posedge();

    FILE *fo = fopen(out, "w");
    FILE *fs = stats ? fopen(stats, "w") : nullptr;
    if (fs) fprintf(fs, "# mode=%s gated=%d groups=%zu\n", mode, (int)gated, groups.size());
    fprintf(fo, "# mode=%s groups=%zu\n", mode, groups.size());
    vluint64_t total_cycles = 0, total_bits_g = 0, total_bits_f = 0;
    for (size_t gi = 0; gi < groups.size(); gi++) {
        Group &g = groups[gi];
        if (g.h < 0 || g.h >= H) { fprintf(stderr, "h=%d out of range H=%d\n", g.h, H); return 1; }
        int max_planes;
        if (is_bf) max_planes = g.e > 0 ? g.e : 1;
        else       max_planes = 23;
        while ((int)g.planes.size() < max_planes) g.planes.push_back(0);

        for (int t = 0; t < 10; t++) {
            dut->in_thr_row[2*t]   = (unsigned)(tau[t][g.h] & 0xffffffffull);
            dut->in_thr_row[2*t+1] = (unsigned)(tau[t][g.h] >> 32);
        }
        dut->in_sop = 1; dut->in_plane_valid = 0;
        dut->in_h = g.h; dut->in_sign = g.sign; dut->in_e = is_bf ? g.e : 23;
        posedge();

        int fed = 0;
        unsigned bits_g = 0, bits_f = 0;
        for (int pi = 0; pi < max_planes; pi++) {
            // 允许 fed==0 提前收：sop 拍粗区间即可判定全锁（zero-plane 组）
            if (is_cert && dut->dbg_locked == 0x3ff) break;
            dut->in_sop = 0; dut->in_plane_valid = 1; dut->in_plane = 0;
            dut->eval();
            unsigned need = (unsigned)(dut->out_need & 0x3ff);
            unsigned send = gated ? need : 0x3ff;
            bits_g += (unsigned)__builtin_popcount(send);
            bits_f += (unsigned)__builtin_popcount(0x3ff);
            dut->in_plane = g.planes[pi] & send;
            posedge();
            fed++;
        }
        if (dut->dbg_locked != 0x3ff) {
            fprintf(stderr, "group %zu not fully locked (fed=%d e=%d)\n", gi, fed, g.e);
            return 2;
        }
        fprintf(fo, "R %03x %d\n", (unsigned)dut->out_dec & 0x3ff, fed);
        if (fs) fprintf(fs, "S %d %u %u\n", fed, bits_g, bits_f);
        total_cycles += 1 + (vluint64_t)fed;
        total_bits_g += bits_g; total_bits_f += bits_f;
        dut->in_sop = 0; dut->in_plane_valid = 0;
    }
    fclose(fo);
    if (fs) fclose(fs);
    double planes = (double)(total_cycles - groups.size()) / (double)groups.size();
    printf("%s gated=%d groups=%zu planes/group=%.4f cycles/group=%.4f "
           "bits/group=%.4f (full %.4f)\n",
           mode, (int)gated, groups.size(), planes, planes + 1.0,
           (double)total_bits_g / (double)groups.size(),
           (double)total_bits_f / (double)groups.size());
    delete dut;
    return 0;
}
