// T5 TB：证书终止门核激励器。自有代码。
// 用法：sim +a=.. +pn=.. +tau_t0=.. .. +tau_t9=.. +stim=stim_fx.txt|stim_bf.txt
//          +mode=fx_full|fx_cert|bf_full|bf_cert +out=result.txt
// 每组输出一行 "R <dec 3hex> <planes>"；dec 为方向折入后的 10 判决。
// cert 模式在 dbg_locked 全 1 后停止供数；full 模式供满全部平面。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "Vcert_gate_core.h"
#include "verilated.h"

static Vcert_gate_core *dut;
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
    dut = new Vcert_gate_core;

    const char *stim = nullptr, *mode = nullptr, *out = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "+stim=", 6)) stim = argv[i] + 6;
        else if (!strncmp(argv[i], "+mode=", 6)) mode = argv[i] + 6;
        else if (!strncmp(argv[i], "+out=", 5)) out = argv[i] + 5;
    }
    if (!stim || !mode || !out) {
        fprintf(stderr, "need +stim= +mode= +out=\n");
        return 1;
    }
    bool is_bf = !strcmp(mode, "bf_full") || !strcmp(mode, "bf_cert");
    bool is_cert = !strcmp(mode, "fx_cert") || !strcmp(mode, "bf_cert");

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
    posedge(); posedge();
    dut->rst_n = 1; posedge();

    FILE *fo = fopen(out, "w");
    fprintf(fo, "# mode=%s groups=%zu\n", mode, groups.size());
    vluint64_t total_cycles = 0;
    for (size_t gi = 0; gi < groups.size(); gi++) {
        Group &g = groups[gi];
        int max_planes;
        if (is_bf) {
            // e=0（全零组）：补 1 个哑平面使检查沿发生
            max_planes = g.e > 0 ? g.e : 1;
        } else {
            // FX：bit23 与符号字冗余不重发（诚实基线 1+23=24 拍/组）
            max_planes = 23;
        }
        if ((int)g.planes.size() < max_planes) {
            // BF e=0 哑平面
            while ((int)g.planes.size() < max_planes) g.planes.push_back(0);
        }
        // sop 拍
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
