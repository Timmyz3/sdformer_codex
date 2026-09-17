// T47 TB：t47_walk_top 的回放。自有代码。
// 同一份 xmem/wmem，同一份时钟，同时驱动零插值行走器与相位分解行走器，
// 把两者写出的 ymem 逐元素与期望文件比较。
// 用法：sim +xm=x.txt +wm=w.txt +exp=e.txt +out=o.txt
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include "Vt47_walk_top.h"
#include "verilated.h"

// 与 RTL 默认参数一致
static const int H = 4, W = 5, CIN = 4;
static const int NX = CIN * H * W;      // 80
static const int NWT = CIN * 9;         // 36
static const int NY = 4 * H * W;        // 80

static Vt47_walk_top *dut;

// Verilator 把 24b 端口给成 uint32，需手工符号扩展
static long long sext24(uint32_t v) {
    return (long long)(int32_t)(v << 8) >> 8;
}

static void posedge() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

static int read_ints(const char *path, std::vector<long> &v, int n) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return 1; }
    v.assign(n, 0);
    char line[256];
    int i = 0;
    while (i < n && fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        v[i++] = strtol(line, nullptr, 10);
    }
    fclose(f);
    if (i != n) { fprintf(stderr, "%s: got %d of %d\n", path, i, n); return 1; }
    return 0;
}

struct RunOut {
    long nslot = 0;
    std::vector<signed long long> y;
};

// 跑一个变体（sel==0 → Z，sel==1 → P），返回写出的 ymem
static RunOut run(int sel, const std::vector<long> &xm, const std::vector<long> &wm) {
    RunOut r;
    r.y.assign(NY, -1);

    dut->rst_n = 0; dut->start_z = 0; dut->start_p = 0;
    dut->zw_data = 0; dut->zx_bit = 0; dut->pw_data = 0; dut->px_bit = 0;
    posedge(); posedge();
    dut->rst_n = 1;
    dut->start_z = (sel == 0); dut->start_p = (sel == 1);
    posedge();
    dut->start_z = 0; dut->start_p = 0;

    for (long c = 0; c < 100000; c++) {
        if (sel == 0) {
            dut->zw_data = (signed char)wm[dut->zw_addr];
            dut->zx_bit  = (xm[dut->zx_addr] != 0);
        } else {
            dut->pw_data = (signed char)wm[dut->pw_addr];
            dut->px_bit  = (xm[dut->px_addr] != 0);
        }
        posedge();
        if (sel == 0) {
            if (dut->zy_we) r.y[dut->zy_addr] = sext24(dut->zy_data);
            if (dut->z_done) { r.nslot = dut->z_nslot; break; }
        } else {
            if (dut->py_we) r.y[dut->py_addr] = sext24(dut->py_data);
            if (dut->p_done) { r.nslot = dut->p_nslot; break; }
        }
    }
    return r;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    const char *fxm = nullptr, *fwm = nullptr, *fexp = nullptr, *fout = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "+xm=", 4)) fxm = argv[i] + 4;
        else if (!strncmp(argv[i], "+wm=", 4)) fwm = argv[i] + 4;
        else if (!strncmp(argv[i], "+exp=", 5)) fexp = argv[i] + 5;
        else if (!strncmp(argv[i], "+out=", 5)) fout = argv[i] + 5;
    }
    if (!fxm || !fwm || !fexp || !fout) {
        fprintf(stderr, "need +xm= +wm= +exp= +out=\n"); return 1;
    }

    std::vector<long> xm, wm, ex;
    if (read_ints(fxm, xm, NX)) return 1;
    if (read_ints(fwm, wm, NWT)) return 1;
    if (read_ints(fexp, ex, NY)) return 1;

    dut = new Vt47_walk_top;

    RunOut z = run(0, xm, wm);
    RunOut p = run(1, xm, wm);

    long zbad = 0, pbad = 0, zp = 0;
    for (int i = 0; i < NY; i++) {
        if (z.y[i] != ex[i]) zbad++;
        if (p.y[i] != ex[i]) pbad++;
        if (z.y[i] != p.y[i]) zp++;
    }

    FILE *fo = fopen(fout, "w");
    fprintf(fo, "Z nslot %ld mismatch %ld\n", z.nslot, zbad);
    fprintf(fo, "P nslot %ld mismatch %ld\n", p.nslot, pbad);
    fprintf(fo, "Z_vs_P mismatch %ld\n", zp);
    for (int i = 0; i < NY; i++)
        fprintf(fo, "%d %lld %lld %lld\n", i, z.y[i], p.y[i], (long long)ex[i]);
    fclose(fo);

    printf("Z nslot=%ld mism=%ld | P nslot=%ld mism=%ld | Z-vs-P mism=%ld\n",
           z.nslot, zbad, p.nslot, pbad, zp);
    delete dut;
    return (zbad == 0 && pbad == 0 && zp == 0) ? 0 : 1;
}
