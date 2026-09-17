// T24 TB：带 VCD 的 BitL 门核激励器（tb_bitl.cpp 的 trace 变体，自有代码）。
// 用途：重放真实激励生成 VCD → Vivado read_vcd → report_power（真实开关活动
// 口径，对齐 FireFly 系 Vivado 报告能耗方法学）。
// 用法：sim +stim=stim_bf.txt +mode=bf_cert +out=result.txt +vcd=dump.vcd [+maxg=N]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "Vcert_gate_bitl.h"
#include "verilated_vcd_c.h"
#include "verilated.h"

static Vcert_gate_bitl *dut;
static VerilatedVcdC *tfp;
static vluint64_t tick_cnt = 0;

static void posedge() {
    dut->clk = 0; dut->eval();
    if (tfp) tfp->dump(tick_cnt * 10);
    dut->clk = 1; dut->eval();
    if (tfp) tfp->dump(tick_cnt * 10 + 5);
    tick_cnt++;
}

struct Group {
    int h, e, sign;
    std::vector<int> planes;
};

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vcert_gate_bitl;

    const char *stim = nullptr, *mode = nullptr, *out = nullptr, *vcd = nullptr;
    long maxg = -1;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "+stim=", 6)) stim = argv[i] + 6;
        else if (!strncmp(argv[i], "+mode=", 6)) mode = argv[i] + 6;
        else if (!strncmp(argv[i], "+out=", 5)) out = argv[i] + 5;
        else if (!strncmp(argv[i], "+vcd=", 5)) vcd = argv[i] + 5;
        else if (!strncmp(argv[i], "+maxg=", 6)) maxg = atol(argv[i] + 6);
    }
    if (!stim || !mode || !out) {
        fprintf(stderr, "need +stim= +mode= +out=\n");
        return 1;
    }
    bool is_bf = !strcmp(mode, "bf_full") || !strcmp(mode, "bf_cert");
    bool is_cert = !strcmp(mode, "fx_cert") || !strcmp(mode, "bf_cert");

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
    if (maxg > 0 && (long)groups.size() > maxg) groups.resize(maxg);

    if (vcd) {
        Verilated::traceEverOn(true);
        tfp = new VerilatedVcdC;
        dut->trace(tfp, 99);
        tfp->open(vcd);
    }

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
        if (is_bf) max_planes = g.e > 0 ? g.e : 1;
        else max_planes = 23;
        if ((int)g.planes.size() < max_planes) {
            while ((int)g.planes.size() < max_planes) g.planes.push_back(0);
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
    if (tfp) { tfp->close(); delete tfp; }
    delete dut;
    return 0;
}
