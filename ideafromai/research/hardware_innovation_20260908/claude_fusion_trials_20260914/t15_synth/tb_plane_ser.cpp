// T15 TB：plane_ser 生产者激励与回放校验。自有代码。
// 从 stim_fx.txt 重建 10 个 signed24 词（二补码：Y[s] = mag − (sign_s ? 2^23 : 0)），
// 喂 plane_ser，捕获 (sign, e, planes)，写成 stim_bf 同格式文件供逐行比对。
// 用法：sim +stim=stim_fx.txt +out=out_bf.txt
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "Vplane_ser.h"
#include "verilated.h"

static Vplane_ser *dut;

static void posedge() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

struct Group {
    int h, sign;
    std::vector<int> planes;
};

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vplane_ser;

    const char *stim = nullptr, *out = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "+stim=", 6)) stim = argv[i] + 6;
        else if (!strncmp(argv[i], "+out=", 5)) out = argv[i] + 5;
    }
    if (!stim || !out) { fprintf(stderr, "need +stim= +out=\n"); return 1; }

    FILE *f = fopen(stim, "r");
    if (!f) { fprintf(stderr, "open %s failed\n", stim); return 1; }
    std::vector<Group> groups;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == 'G') {
            Group g;
            int sign = 0;
            if (sscanf(line, "G %d %x", &g.h, &sign) < 2) return 1;
            g.sign = sign & 0x3ff;
            groups.push_back(g);
        } else if (line[0] == 'B') {
            unsigned p;
            sscanf(line, "B %x", &p);
            groups.back().planes.push_back(p & 0x3ff);
        }
    }
    fclose(f);

    dut->rst_n = 0; dut->in_sop = 0; dut->in_plane_req = 0;
    for (int w = 0; w < 8; w++) dut->in_ywords[w] = 0;
    posedge(); posedge();
    dut->rst_n = 1; posedge();

    FILE *fo = fopen(out, "w");
    for (size_t gi = 0; gi < groups.size(); gi++) {
        Group &g = groups[gi];
        while (g.planes.size() < 23) g.planes.push_back(0);
        long long ywords[10];
        for (int s = 0; s < 10; s++) {
            int mag = 0;
            for (int i = 0; i < 23; i++)
                mag = (mag << 1) | ((g.planes[i] >> s) & 1);
            ywords[s] = (long long)mag - (((g.sign >> s) & 1) << 23);
        }
        // 打包 {Y9..Y0}，各 24b（240b 端口 → Verilator WData 32bit×8）
        unsigned words[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int s = 0; s < 10; s++) {
            unsigned y = (unsigned)(ywords[s] & 0xffffff);
            int base = s * 24;
            for (int b = 0; b < 24; b++)
                if ((y >> b) & 1u)
                    words[(base + b) / 32] |= 1u << ((base + b) % 32);
        }
        for (int w = 0; w < 8; w++)
            dut->in_ywords[w] = words[w];
        dut->in_sop = 1; dut->in_plane_req = 0;
        posedge();                       // 该沿锁存 mag/sign/e，下一拍 out_sop_valid
        dut->in_sop = 0;
        if (!dut->out_sop_valid) { fprintf(stderr, "g%zu no sop_valid\n", gi); return 2; }
        unsigned sign_o = dut->out_sign & 0x3ff;
        int e = dut->out_e;
        fprintf(fo, "G %d %03x %d\n", g.h, sign_o, e);
        for (int k = 0; k < e; k++) {
            dut->in_plane_req = 1;
            posedge();
            dut->in_plane_req = 0;
            if (!dut->out_plane_valid) { fprintf(stderr, "g%zu plane %d missing\n", gi, k); return 2; }
            fprintf(fo, "B %03x\n", dut->out_plane & 0x3ff);
        }
        if (!dut->out_done) { fprintf(stderr, "g%zu no done (e=%d)\n", gi, e); return 2; }
    }
    fclose(fo);
    printf("plane_ser replay groups=%zu\n", groups.size());
    delete dut;
    return 0;
}
