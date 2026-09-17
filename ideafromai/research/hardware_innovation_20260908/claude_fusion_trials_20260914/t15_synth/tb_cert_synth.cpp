// T15 TB：cert_gate_bitl_synth（综合口径变体）激励器。自有代码，派生自 t10_rtl/tb_bitl.cpp。
// 差异：tau ROM → in_thr_row 640b 端口，TB 从 results/t5_rtl/<trace>/tau_t%d.hex 装载，
// sop 拍随 h 送入对应行；其余与 T10 TB 逻辑一致（4 模式，cert 模式锁定即停）。
// 用法：sim +stim=.. +mode=fx_full|fx_cert|bf_full|bf_cert +taudir=.. +out=result.txt
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "Vcert_gate_bitl_synth.h"
#include "verilated.h"

static Vcert_gate_bitl_synth *dut;
static vluint64_t tick_cnt = 0;

static void posedge() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
    tick_cnt++;
}

struct Group {
    int h, e, sign;
    std::vector<int> planes;  // MSB-first 数据平面
};

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vcert_gate_bitl_synth;

    const char *stim = nullptr, *mode = nullptr, *out = nullptr, *taudir = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "+stim=", 6)) stim = argv[i] + 6;
        else if (!strncmp(argv[i], "+mode=", 6)) mode = argv[i] + 6;
        else if (!strncmp(argv[i], "+out=", 5)) out = argv[i] + 5;
        else if (!strncmp(argv[i], "+taudir=", 8)) taudir = argv[i] + 8;
    }
    if (!stim || !mode || !out || !taudir) {
        fprintf(stderr, "need +stim= +mode= +out= +taudir=\n");
        return 1;
    }
    bool is_bf = !strcmp(mode, "bf_full") || !strcmp(mode, "bf_cert");
    bool is_cert = !strcmp(mode, "fx_cert") || !strcmp(mode, "bf_cert");

    // 装载 10 个 tau 行表：tau[t][h]
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

    // 解析激励
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

    // 复位
    dut->rst_n = 0; dut->in_sop = 0; dut->in_plane_valid = 0;
    dut->in_h = 0; dut->in_sign = 0; dut->in_e = 0; dut->in_plane = 0;
    for (int w = 0; w < 20; w++) dut->in_thr_row[w] = 0;
    posedge(); posedge();
    dut->rst_n = 1; posedge();

    FILE *fo = fopen(out, "w");
    fprintf(fo, "# mode=%s groups=%zu\n", mode, groups.size());
    vluint64_t total_cycles = 0;
    for (size_t gi = 0; gi < groups.size(); gi++) {
        Group &g = groups[gi];
        if (g.h < 0 || g.h >= H) { fprintf(stderr, "h=%d out of range H=%d\n", g.h, H); return 1; }
        int max_planes;
        if (is_bf) {
            max_planes = g.e > 0 ? g.e : 1;   // e=0 全零组：补哑平面使检查沿发生
        } else {
            max_planes = 23;                   // FX：bit23 与符号字冗余不重发
        }
        if ((int)g.planes.size() < max_planes) {
            while ((int)g.planes.size() < max_planes) g.planes.push_back(0);
        }
        // sop 拍：h/sign/e + thr 行
        for (int t = 0; t < 10; t++) {
            dut->in_thr_row[2*t]   = (unsigned)(tau[t][g.h] & 0xffffffffull);
            dut->in_thr_row[2*t+1] = (unsigned)(tau[t][g.h] >> 32);
        }
        dut->in_sop = 1; dut->in_plane_valid = 0;
        dut->in_h = g.h; dut->in_sign = g.sign; dut->in_e = is_bf ? g.e : 23;
        posedge();
        int fed = 0;
        for (int pi = 0; pi < max_planes; pi++) {
            if (is_cert && fed > 0 && dut->dbg_locked == 0x3ff) break;
            dut->in_sop = 0; dut->in_plane_valid = 1;
            dut->in_plane = g.planes[pi];
            posedge();
            fed++;
        }
        if (dut->dbg_locked != 0x3ff) {
            fprintf(stderr, "group %zu not fully locked (fed=%d e=%d)\n", gi, fed, g.e);
            return 2;
        }
        fprintf(fo, "R %03x %d\n", (unsigned)dut->out_dec & 0x3ff, fed);
        total_cycles += 1 + (vluint64_t)fed;
        dut->in_sop = 0; dut->in_plane_valid = 0;
    }
    fclose(fo);
    printf("%s groups=%zu cycles/group=%.4f\n", mode, groups.size(),
           (double)total_cycles / (double)groups.size());
    delete dut;
    return 0;
}
