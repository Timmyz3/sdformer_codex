/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP3
// Date      : Thu Aug 27 14:59:19 2026
/////////////////////////////////////////////////////////////


module m514_c2_convtranspose_k3s2_polyphase_address_mapper ( clk_core, 
        rst_core, event_valid, event_ready, event_tag, event_time, 
        event_source_channel, event_source_y, event_source_x, 
        event_input_height, event_input_width, event_last, event_accept, 
        tap_valid, tap_ready, tap_tag, tap_time, tap_source_channel, 
        tap_source_y, tap_source_x, tap_kernel_y, tap_kernel_x, 
        tap_kernel_index, tap_destination_y, tap_destination_x, tap_phase_bank, 
        tap_last_for_event, tap_stream_last, tap_accept, protocol_error, busy
 );
  input [23:0] event_tag;
  input [3:0] event_time;
  input [11:0] event_source_channel;
  input [9:0] event_source_y;
  input [9:0] event_source_x;
  input [9:0] event_input_height;
  input [9:0] event_input_width;
  output [23:0] tap_tag;
  output [3:0] tap_time;
  output [11:0] tap_source_channel;
  output [9:0] tap_source_y;
  output [9:0] tap_source_x;
  output [1:0] tap_kernel_y;
  output [1:0] tap_kernel_x;
  output [3:0] tap_kernel_index;
  output [9:0] tap_destination_y;
  output [9:0] tap_destination_x;
  output [1:0] tap_phase_bank;
  input clk_core, rst_core, event_valid, event_last, tap_ready;
  output event_ready, event_accept, tap_valid, tap_last_for_event,
         tap_stream_last, tap_accept, protocol_error, busy;
  wire   stream_last_q, N98, N125, C39_DATA2_2, C39_DATA2_3, C39_DATA2_4,
         C39_DATA2_5, C39_DATA2_6, C39_DATA2_7, C39_DATA2_8, C38_DATA2_2,
         C38_DATA2_3, C38_DATA2_4, C38_DATA2_5, C38_DATA2_6, C38_DATA2_7,
         C38_DATA2_8, n121, n122, n123, n124, n125, n126, n127, n128, n129,
         n131, n132, n133, n134, n135, n136, n137, n138, n139, n141, n142,
         n143, n144, n145, n146, n147, n148, n149, n150, n151, n152, n153,
         n154, n155, n156, n157, n158, n159, n160, n161, n162, n163, n164,
         n165, n166, n167, n168, n169, n170, n171, n172, n173, n174, n175,
         n176, n177, n178, n179, n180, n183, n184, n185, n186, n187, n188,
         n189, n190, n191, n192, DP_OP_67J1_123_5569_n8,
         DP_OP_67J1_123_5569_n7, DP_OP_67J1_123_5569_n6,
         DP_OP_67J1_123_5569_n5, DP_OP_67J1_123_5569_n4,
         DP_OP_67J1_123_5569_n3, DP_OP_67J1_123_5569_n2,
         DP_OP_70J1_126_5313_n8, DP_OP_70J1_126_5313_n7,
         DP_OP_70J1_126_5313_n6, DP_OP_70J1_126_5313_n5,
         DP_OP_70J1_126_5313_n4, DP_OP_70J1_126_5313_n3,
         DP_OP_70J1_126_5313_n2, n197, n198, n199, n200, n201, n202, n203,
         n204, n205, n206, n207, n208, n209, n210, n211, n212, n213, n214,
         n215, n216, n217, n218, n219, n220, n221, n222, n223, n224, n225,
         n226, n227, n228, n229, n230, n231, n232, n233, n234, n235, n236,
         n237, n238, n239, n240, n241, n242, n243, n244, n245, n246, n247,
         n248, n249, n250, n251, n252, n253, n254, n255, n256, n258, n259,
         n260, n261, n262, n264, n265, n266, n268, n269, n270, n271, n272,
         n273, n274, n275, n276, n278, n279, n280, n281, n282, n283, n284,
         n285, n286, n287, n288, n290, n291, n292, n293, n294, n295, n296,
         n297, n298, n299, n300, n301, n302, n303, n304, n305, n306, n307,
         n308, n311, n312, n322, n323, n324, n325, n326, n327, n328, n331,
         n332, n333, n334, n335, n336, n337, n338, n339, n340, n341, n342,
         n343, n344, n345, n346, n347, n348, n349, n350, n351, n352, n353,
         n354, n355, n356, n357, n358, n359, n360, n361, n362, n363, n364,
         n365, n366, n367, n368, n369, n370, n371, n372, n373, n374, n375,
         n376, n377, n378, n379, n380, n381, n382, n383, n384, n385, n386,
         n387, n388, n389, n390, n391, n392, n393, n394, n395, n396, n397,
         n398, n399, n400, n401, n402, n403, n404, n405, n406, n407, n408,
         n409, n410, n411, n412, n413, n414, n415, n416, n417, n418, n419,
         n420, n421, n422, n423, n424, n425, n426, n427, n428, n429, n430,
         n431, n432, n433, n434, n435, n436, n437, n438, n439, n440, n441,
         n442, n443, n444, n445, n446, n447, n448, n449, n450, n451, n452,
         n453, n454, n455, n456, n457, n458, n459, n460, n461, n462, n463,
         n464, n465;
  wire   [8:0] pending_q;

  FA1D0BWP35P140 DP_OP_70J1_126_5313_U9 ( .A(tap_source_x[1]), .B(N125), .CI(
        n311), .CO(DP_OP_70J1_126_5313_n8), .S(C39_DATA2_2) );
  FA1D0BWP35P140 DP_OP_70J1_126_5313_U8 ( .A(tap_source_x[2]), .B(N125), .CI(
        DP_OP_70J1_126_5313_n8), .CO(DP_OP_70J1_126_5313_n7), .S(C39_DATA2_3)
         );
  FA1D0BWP35P140 DP_OP_70J1_126_5313_U7 ( .A(tap_source_x[3]), .B(N125), .CI(
        DP_OP_70J1_126_5313_n7), .CO(DP_OP_70J1_126_5313_n6), .S(C39_DATA2_4)
         );
  FA1D0BWP35P140 DP_OP_70J1_126_5313_U6 ( .A(tap_source_x[4]), .B(N125), .CI(
        DP_OP_70J1_126_5313_n6), .CO(DP_OP_70J1_126_5313_n5), .S(C39_DATA2_5)
         );
  FA1D0BWP35P140 DP_OP_70J1_126_5313_U5 ( .A(tap_source_x[5]), .B(N125), .CI(
        DP_OP_70J1_126_5313_n5), .CO(DP_OP_70J1_126_5313_n4), .S(C39_DATA2_6)
         );
  FA1D0BWP35P140 DP_OP_70J1_126_5313_U4 ( .A(tap_source_x[6]), .B(N125), .CI(
        DP_OP_70J1_126_5313_n4), .CO(DP_OP_70J1_126_5313_n3), .S(C39_DATA2_7)
         );
  FA1D0BWP35P140 DP_OP_70J1_126_5313_U3 ( .A(tap_source_x[7]), .B(N125), .CI(
        DP_OP_70J1_126_5313_n3), .CO(DP_OP_70J1_126_5313_n2), .S(C39_DATA2_8)
         );
  FA1D0BWP35P140 DP_OP_67J1_123_5569_U9 ( .A(tap_source_y[1]), .B(N98), .CI(
        n312), .CO(DP_OP_67J1_123_5569_n8), .S(C38_DATA2_2) );
  FA1D0BWP35P140 DP_OP_67J1_123_5569_U8 ( .A(tap_source_y[2]), .B(N98), .CI(
        DP_OP_67J1_123_5569_n8), .CO(DP_OP_67J1_123_5569_n7), .S(C38_DATA2_3)
         );
  FA1D0BWP35P140 DP_OP_67J1_123_5569_U7 ( .A(tap_source_y[3]), .B(N98), .CI(
        DP_OP_67J1_123_5569_n7), .CO(DP_OP_67J1_123_5569_n6), .S(C38_DATA2_4)
         );
  FA1D0BWP35P140 DP_OP_67J1_123_5569_U6 ( .A(tap_source_y[4]), .B(N98), .CI(
        DP_OP_67J1_123_5569_n6), .CO(DP_OP_67J1_123_5569_n5), .S(C38_DATA2_5)
         );
  FA1D0BWP35P140 DP_OP_67J1_123_5569_U5 ( .A(tap_source_y[5]), .B(N98), .CI(
        DP_OP_67J1_123_5569_n5), .CO(DP_OP_67J1_123_5569_n4), .S(C38_DATA2_6)
         );
  FA1D0BWP35P140 DP_OP_67J1_123_5569_U4 ( .A(tap_source_y[6]), .B(N98), .CI(
        DP_OP_67J1_123_5569_n4), .CO(DP_OP_67J1_123_5569_n3), .S(C38_DATA2_7)
         );
  FA1D0BWP35P140 DP_OP_67J1_123_5569_U3 ( .A(tap_source_y[7]), .B(N98), .CI(
        DP_OP_67J1_123_5569_n3), .CO(DP_OP_67J1_123_5569_n2), .S(C38_DATA2_8)
         );
  CKND0BWP35P140 U231 ( .I(n255), .ZN(event_accept) );
  ND2D0BWP35P140 U232 ( .A1(n266), .A2(event_accept), .ZN(n276) );
  CKND2D1BWP35P140 U233 ( .A1(event_ready), .A2(event_valid), .ZN(n255) );
  OAI211D0BWP35P140 U234 ( .A1(n271), .A2(n325), .B(tap_valid), .C(n249), .ZN(
        n250) );
  OAI211D0BWP35P140 U235 ( .A1(event_input_height[9]), .A2(n239), .B(n238), 
        .C(n237), .ZN(n253) );
  OAI21D0BWP35P140 U236 ( .A1(n236), .A2(event_input_width[9]), .B(n235), .ZN(
        n237) );
  CKND2D1BWP35P140 U237 ( .A1(n271), .A2(n199), .ZN(n259) );
  MAOI222D0BWP35P140 U238 ( .A(event_source_y[8]), .B(n214), .C(n213), .ZN(
        n239) );
  MAOI222D0BWP35P140 U239 ( .A(event_input_width[8]), .B(n231), .C(n284), .ZN(
        n236) );
  MAOI222D0BWP35P140 U240 ( .A(event_source_x[7]), .B(n230), .C(n229), .ZN(
        n231) );
  MAOI222D0BWP35P140 U241 ( .A(event_input_height[7]), .B(n212), .C(n211), 
        .ZN(n214) );
  CKND2D1BWP35P140 U242 ( .A1(n283), .A2(n265), .ZN(n307) );
  AOI22D0BWP35P140 U243 ( .A1(pending_q[7]), .A2(n265), .B1(pending_q[4]), 
        .B2(n248), .ZN(n249) );
  MAOI222D0BWP35P140 U244 ( .A(event_input_width[6]), .B(n286), .C(n228), .ZN(
        n229) );
  MAOI222D0BWP35P140 U245 ( .A(event_source_y[6]), .B(n210), .C(n209), .ZN(
        n212) );
  OAI21D0BWP35P140 U247 ( .A1(n247), .A2(n322), .B(n246), .ZN(n251) );
  MAOI222D0BWP35P140 U248 ( .A(event_input_height[5]), .B(n208), .C(n272), 
        .ZN(n210) );
  MAOI222D0BWP35P140 U249 ( .A(event_source_x[5]), .B(n227), .C(n226), .ZN(
        n228) );
  MAOI222D0BWP35P140 U250 ( .A(event_input_width[4]), .B(n285), .C(n225), .ZN(
        n226) );
  MAOI222D0BWP35P140 U251 ( .A(event_source_y[4]), .B(n207), .C(n206), .ZN(
        n208) );
  AOI22D0BWP35P140 U252 ( .A1(pending_q[3]), .A2(n256), .B1(n291), .B2(n245), 
        .ZN(n246) );
  OA21D0BWP35P140 U253 ( .A1(n291), .A2(n292), .B(n256), .Z(n197) );
  OAI21D0BWP35P140 U255 ( .A1(pending_q[5]), .A2(pending_q[4]), .B(n271), .ZN(
        n201) );
  MAOI222D0BWP35P140 U256 ( .A(event_source_x[3]), .B(n224), .C(n223), .ZN(
        n225) );
  MAOI222D0BWP35P140 U257 ( .A(event_input_height[3]), .B(n205), .C(n273), 
        .ZN(n207) );
  MAOI222D0BWP35P140 U258 ( .A(event_source_y[2]), .B(n204), .C(n203), .ZN(
        n205) );
  MAOI222D0BWP35P140 U259 ( .A(event_input_width[2]), .B(n222), .C(n221), .ZN(
        n223) );
  CKND2D1BWP35P140 U260 ( .A1(n244), .A2(n302), .ZN(n291) );
  OAI21D0BWP35P140 U261 ( .A1(n244), .A2(n302), .B(n292), .ZN(n245) );
  MOAI22D0BWP35P140 U262 ( .A1(n220), .A2(event_source_x[0]), .B1(n219), .B2(
        event_input_width[1]), .ZN(n221) );
  MAOI22D0BWP35P140 U263 ( .A1(event_input_height[0]), .A2(n202), .B1(n215), 
        .B2(event_source_y[1]), .ZN(n204) );
  AOI211D0BWP35P140 U264 ( .A1(event_input_height[9]), .A2(n218), .B(
        event_source_y[9]), .C(event_source_x[9]), .ZN(n238) );
  CKND0BWP35P140 U265 ( .I(pending_q[8]), .ZN(n260) );
  OAI21D0BWP35P140 U266 ( .A1(n219), .A2(event_input_width[1]), .B(
        event_input_width[0]), .ZN(n220) );
  DEL025D1BWP35P140 U270 ( .I(tap_destination_x[0]), .Z(tap_phase_bank[0]) );
  CKND0BWP35P140 U271 ( .I(pending_q[0]), .ZN(n244) );
  CKND0BWP35P140 U272 ( .I(pending_q[1]), .ZN(n302) );
  NR3D0P7BWP35P140 U273 ( .A1(n291), .A2(pending_q[2]), .A3(pending_q[3]), 
        .ZN(n271) );
  CKND0BWP35P140 U278 ( .I(n201), .ZN(n198) );
  NR2D1BWP35P140 U280 ( .A1(n259), .A2(n260), .ZN(n242) );
  IND2D1BWP35P140 U281 ( .A1(n242), .B1(n201), .ZN(tap_kernel_x[0]) );
  CKND0BWP35P140 U282 ( .I(tap_kernel_x[0]), .ZN(tap_destination_x[0]) );
  OR2D1BWP35P140 U283 ( .A1(n242), .A2(n307), .Z(tap_kernel_y[0]) );
  CKND0BWP35P140 U284 ( .I(tap_kernel_y[0]), .ZN(tap_phase_bank[1]) );
  DEL025D1BWP35P140 U285 ( .I(tap_phase_bank[1]), .Z(tap_destination_y[0]) );
  CKND0BWP35P140 U286 ( .I(n199), .ZN(tap_kernel_index[0]) );
  ND3D1BWP35P140 U288 ( .A1(n366), .A2(n271), .A3(n323), .ZN(n262) );
  CKND0BWP35P140 U289 ( .I(pending_q[2]), .ZN(n292) );
  IND3D1BWP35P140 U290 ( .A1(n291), .B1(pending_q[3]), .B2(n292), .ZN(n256) );
  ND3D1BWP35P140 U291 ( .A1(n265), .A2(n262), .A3(n256), .ZN(n200) );
  AOI21D0BWP35P140 U292 ( .A1(pending_q[1]), .A2(n244), .B(n200), .ZN(n241) );
  OAI21D0BWP35P140 U293 ( .A1(n201), .A2(n241), .B(n197), .ZN(tap_kernel_y[1])
         );
  NR2D0BWP35P140 U294 ( .A1(tap_kernel_y[0]), .A2(tap_kernel_y[1]), .ZN(N98)
         );
  XOR2UD0BWP35P140 U295 ( .A1(N98), .A2(tap_source_y[0]), .Z(
        tap_destination_y[1]) );
  CKND0BWP35P140 U296 ( .I(event_input_height[1]), .ZN(n215) );
  AOI21D0BWP35P140 U297 ( .A1(event_source_y[1]), .A2(n215), .B(
        event_source_y[0]), .ZN(n202) );
  CKND0BWP35P140 U298 ( .I(event_input_height[2]), .ZN(n203) );
  CKND0BWP35P140 U299 ( .I(event_source_y[3]), .ZN(n273) );
  CKND0BWP35P140 U300 ( .I(event_input_height[4]), .ZN(n206) );
  CKND0BWP35P140 U301 ( .I(event_source_y[5]), .ZN(n272) );
  CKND0BWP35P140 U302 ( .I(event_input_height[6]), .ZN(n209) );
  CKND0BWP35P140 U303 ( .I(event_source_y[7]), .ZN(n211) );
  CKND0BWP35P140 U304 ( .I(event_input_height[8]), .ZN(n213) );
  NR3D0P7BWP35P140 U305 ( .A1(event_input_height[6]), .A2(
        event_input_height[5]), .A3(event_input_height[3]), .ZN(n217) );
  NR4D0BWP35P140 U306 ( .A1(event_input_height[8]), .A2(event_input_height[4]), 
        .A3(event_input_height[2]), .A4(event_input_height[0]), .ZN(n216) );
  IND4D1BWP35P140 U307 ( .A1(event_input_height[7]), .B1(n217), .B2(n216), 
        .B3(n215), .ZN(n218) );
  CKND0BWP35P140 U308 ( .I(event_input_width[7]), .ZN(n230) );
  CKND0BWP35P140 U309 ( .I(event_source_x[6]), .ZN(n286) );
  CKND0BWP35P140 U310 ( .I(event_input_width[5]), .ZN(n227) );
  CKND0BWP35P140 U311 ( .I(event_source_x[4]), .ZN(n285) );
  CKND0BWP35P140 U312 ( .I(event_input_width[3]), .ZN(n224) );
  CKND0BWP35P140 U313 ( .I(event_source_x[2]), .ZN(n222) );
  CKND0BWP35P140 U314 ( .I(event_source_x[1]), .ZN(n219) );
  CKND0BWP35P140 U315 ( .I(event_source_x[8]), .ZN(n284) );
  OR4D1BWP35P140 U317 ( .A1(event_input_width[7]), .A2(event_input_width[2]), 
        .A3(event_input_width[5]), .A4(event_input_width[4]), .Z(n232) );
  NR4D0BWP35P140 U318 ( .A1(event_input_width[1]), .A2(event_input_width[3]), 
        .A3(event_input_width[0]), .A4(n232), .ZN(n233) );
  AOI21D0BWP35P140 U320 ( .A1(event_valid), .A2(n253), .B(protocol_error), 
        .ZN(n240) );
  MUX2D0BWP35P140 U322 ( .I0(tap_source_y[1]), .I1(C38_DATA2_2), .S(
        tap_phase_bank[1]), .Z(tap_destination_y[2]) );
  CKND0BWP35P140 U324 ( .I(tap_kernel_y[1]), .ZN(n305) );
  AOI21D0BWP35P140 U325 ( .A1(n307), .A2(tap_kernel_x[1]), .B(n242), .ZN(n243)
         );
  OAI21D0BWP35P140 U326 ( .A1(n305), .A2(tap_kernel_x[1]), .B(n243), .ZN(
        tap_kernel_index[2]) );
  CKND0BWP35P140 U327 ( .I(n256), .ZN(tap_kernel_index[3]) );
  MUX2D0BWP35P140 U328 ( .I0(tap_source_y[2]), .I1(C38_DATA2_3), .S(
        tap_phase_bank[1]), .Z(tap_destination_y[3]) );
  NR2D0BWP35P140 U329 ( .A1(tap_kernel_x[1]), .A2(tap_kernel_x[0]), .ZN(N125)
         );
  XOR2UD0BWP35P140 U330 ( .A1(N125), .A2(tap_source_x[0]), .Z(
        tap_destination_x[1]) );
  MUX2D0BWP35P140 U331 ( .I0(tap_source_y[3]), .I1(C38_DATA2_4), .S(
        tap_phase_bank[1]), .Z(tap_destination_y[4]) );
  MUX2D0BWP35P140 U332 ( .I0(tap_source_x[1]), .I1(C39_DATA2_2), .S(
        tap_destination_x[0]), .Z(tap_destination_x[2]) );
  MUX2D0BWP35P140 U333 ( .I0(tap_source_y[4]), .I1(C38_DATA2_5), .S(
        tap_phase_bank[1]), .Z(tap_destination_y[5]) );
  MUX2D0BWP35P140 U334 ( .I0(tap_source_y[5]), .I1(C38_DATA2_6), .S(
        tap_phase_bank[1]), .Z(tap_destination_y[6]) );
  MUX2D0BWP35P140 U335 ( .I0(tap_source_x[2]), .I1(C39_DATA2_3), .S(
        tap_destination_x[0]), .Z(tap_destination_x[3]) );
  OA21D0BWP35P140 U336 ( .A1(pending_q[8]), .A2(n259), .B(busy), .Z(tap_valid)
         );
  AOI211D1BWP35P140 U337 ( .A1(pending_q[8]), .A2(n259), .B(n251), .C(n250), 
        .ZN(tap_last_for_event) );
  AOI21D0BWP35P140 U339 ( .A1(tap_last_for_event), .A2(tap_ready), .B(n465), 
        .ZN(n254) );
  AN2D0BWP35P140 U342 ( .A1(tap_last_for_event), .A2(stream_last_q), .Z(
        tap_stream_last) );
  MUX2D0BWP35P140 U343 ( .I0(tap_source_y[6]), .I1(C38_DATA2_7), .S(
        tap_phase_bank[1]), .Z(tap_destination_y[7]) );
  CKND0BWP35P140 U345 ( .I(rst_core), .ZN(n266) );
  ND2D0BWP35P140 U346 ( .A1(n255), .A2(n266), .ZN(n279) );
  NR2D0BWP35P140 U347 ( .A1(tap_accept), .A2(n279), .ZN(n299) );
  AOI21D0BWP35P140 U348 ( .A1(n266), .A2(n256), .B(n299), .ZN(n258) );
  OAI21D0BWP35P140 U350 ( .A1(n258), .A2(n363), .B(n276), .ZN(n185) );
  AOI21D0BWP35P140 U351 ( .A1(n266), .A2(n259), .B(n299), .ZN(n261) );
  OAI21D0BWP35P140 U352 ( .A1(n261), .A2(n378), .B(n276), .ZN(n190) );
  AOI21D0BWP35P140 U353 ( .A1(n266), .A2(n262), .B(n299), .ZN(n264) );
  OAI21D0BWP35P140 U354 ( .A1(n264), .A2(n325), .B(n276), .ZN(n187) );
  AOI21D0BWP35P140 U355 ( .A1(n266), .A2(n265), .B(n299), .ZN(n268) );
  MUX2D0BWP35P140 U358 ( .I0(tap_source_x[3]), .I1(C39_DATA2_4), .S(
        tap_destination_x[0]), .Z(tap_destination_x[4]) );
  MUX2D0BWP35P140 U359 ( .I0(tap_source_y[7]), .I1(C38_DATA2_8), .S(
        tap_phase_bank[1]), .Z(tap_destination_y[8]) );
  XOR2UD0BWP35P140 U360 ( .A1(N98), .A2(tap_source_y[8]), .Z(n269) );
  XOR2UD0BWP35P140 U361 ( .A1(n269), .A2(DP_OP_67J1_123_5569_n2), .Z(n270) );
  MUX2D0BWP35P140 U362 ( .I0(n270), .I1(tap_source_y[8]), .S(tap_kernel_y[0]), 
        .Z(tap_destination_y[9]) );
  AN2D0BWP35P140 U363 ( .A1(N98), .A2(tap_source_y[0]), .Z(n312) );
  MUX2D0BWP35P140 U364 ( .I0(tap_source_x[4]), .I1(C39_DATA2_5), .S(
        tap_destination_x[0]), .Z(tap_destination_x[5]) );
  IAO21D1BWP35P140 U365 ( .A1(n279), .A2(n271), .B(n299), .ZN(n278) );
  NR3D0BWP35P140 U366 ( .A1(event_source_y[8]), .A2(event_source_y[6]), .A3(
        event_source_y[7]), .ZN(n275) );
  NR4D0BWP35P140 U367 ( .A1(event_source_y[4]), .A2(event_source_y[2]), .A3(
        event_source_y[1]), .A4(event_source_y[0]), .ZN(n274) );
  ND4D0BWP35P140 U368 ( .A1(n275), .A2(n274), .A3(n273), .A4(n272), .ZN(n298)
         );
  CKND0BWP35P140 U369 ( .I(n276), .ZN(n281) );
  ND2D0BWP35P140 U370 ( .A1(n298), .A2(n281), .ZN(n301) );
  OAI21D0BWP35P140 U371 ( .A1(n278), .A2(n350), .B(n301), .ZN(n186) );
  MUX2D0BWP35P140 U372 ( .I0(tap_source_x[5]), .I1(C39_DATA2_6), .S(
        tap_destination_x[0]), .Z(tap_destination_x[6]) );
  CKND0BWP35P140 U373 ( .I(n279), .ZN(n282) );
  CKND0BWP35P140 U374 ( .I(n279), .ZN(n300) );
  AO22D0BWP35P140 U375 ( .A1(n281), .A2(event_last), .B1(n300), .B2(n357), .Z(
        n191) );
  AO22D0BWP35P140 U376 ( .A1(n281), .A2(event_tag[23]), .B1(n282), .B2(n428), 
        .Z(n180) );
  AO22D0BWP35P140 U377 ( .A1(n281), .A2(event_tag[22]), .B1(n282), .B2(n429), 
        .Z(n179) );
  AO22D0BWP35P140 U378 ( .A1(n281), .A2(event_tag[21]), .B1(n282), .B2(n359), 
        .Z(n178) );
  AO22D0BWP35P140 U379 ( .A1(n281), .A2(event_tag[20]), .B1(n282), .B2(n430), 
        .Z(n177) );
  AO22D0BWP35P140 U380 ( .A1(n281), .A2(event_tag[19]), .B1(n282), .B2(n431), 
        .Z(n176) );
  AO22D0BWP35P140 U381 ( .A1(n281), .A2(event_tag[18]), .B1(n282), .B2(n432), 
        .Z(n175) );
  AO22D0BWP35P140 U382 ( .A1(n281), .A2(event_tag[16]), .B1(n282), .B2(n433), 
        .Z(n173) );
  AO22D0BWP35P140 U383 ( .A1(n281), .A2(event_tag[15]), .B1(n282), .B2(n434), 
        .Z(n172) );
  AO22D0BWP35P140 U384 ( .A1(n281), .A2(event_tag[14]), .B1(n282), .B2(n435), 
        .Z(n171) );
  AO22D0BWP35P140 U385 ( .A1(n281), .A2(event_source_channel[8]), .B1(n300), 
        .B2(n454), .Z(n149) );
  AO22D0BWP35P140 U386 ( .A1(n281), .A2(event_tag[13]), .B1(n282), .B2(n436), 
        .Z(n170) );
  AO22D0BWP35P140 U387 ( .A1(n281), .A2(event_source_channel[7]), .B1(n300), 
        .B2(n353), .Z(n148) );
  AO22D0BWP35P140 U388 ( .A1(n281), .A2(event_source_channel[6]), .B1(n300), 
        .B2(n455), .Z(n147) );
  AO22D0BWP35P140 U389 ( .A1(n281), .A2(event_source_channel[5]), .B1(n300), 
        .B2(n456), .Z(n146) );
  AO22D0BWP35P140 U390 ( .A1(n281), .A2(event_source_channel[4]), .B1(n300), 
        .B2(n457), .Z(n145) );
  AO22D0BWP35P140 U391 ( .A1(n281), .A2(event_source_channel[3]), .B1(n300), 
        .B2(n458), .Z(n144) );
  AO22D0BWP35P140 U392 ( .A1(n281), .A2(event_source_channel[2]), .B1(n300), 
        .B2(n459), .Z(n143) );
  AO22D0BWP35P140 U393 ( .A1(n281), .A2(event_source_y[2]), .B1(
        tap_source_y[2]), .B2(n282), .Z(n133) );
  AO22D0BWP35P140 U394 ( .A1(n281), .A2(event_source_y[1]), .B1(
        tap_source_y[1]), .B2(n282), .Z(n132) );
  AO22D0BWP35P140 U395 ( .A1(n281), .A2(event_source_y[0]), .B1(n450), .B2(
        n282), .Z(n131) );
  AO22D0BWP35P140 U396 ( .A1(n281), .A2(event_source_channel[1]), .B1(n300), 
        .B2(n460), .Z(n142) );
  AO22D0BWP35P140 U397 ( .A1(n281), .A2(event_source_channel[0]), .B1(n300), 
        .B2(n461), .Z(n141) );
  AO22D0BWP35P140 U398 ( .A1(n281), .A2(event_source_channel[10]), .B1(n300), 
        .B2(n452), .Z(n151) );
  AO22D0BWP35P140 U399 ( .A1(n281), .A2(event_source_channel[9]), .B1(n300), 
        .B2(n453), .Z(n150) );
  AO22D0BWP35P140 U400 ( .A1(n281), .A2(event_source_y[4]), .B1(
        tap_source_y[4]), .B2(n300), .Z(n135) );
  AO22D0BWP35P140 U401 ( .A1(n281), .A2(event_source_y[5]), .B1(
        tap_source_y[5]), .B2(n300), .Z(n136) );
  AO22D0BWP35P140 U402 ( .A1(n281), .A2(event_source_channel[11]), .B1(n300), 
        .B2(n451), .Z(n152) );
  AO22D0BWP35P140 U403 ( .A1(n281), .A2(event_source_y[3]), .B1(
        tap_source_y[3]), .B2(n282), .Z(n134) );
  CKND0BWP35P140 U404 ( .I(n276), .ZN(n280) );
  AO22D0BWP35P140 U405 ( .A1(n280), .A2(event_source_x[0]), .B1(n427), .B2(
        n282), .Z(n121) );
  AO22D0BWP35P140 U406 ( .A1(n280), .A2(event_source_y[8]), .B1(n398), .B2(
        n282), .Z(n139) );
  AO22D0BWP35P140 U407 ( .A1(n280), .A2(event_source_x[8]), .B1(n405), .B2(
        n282), .Z(n129) );
  AO22D0BWP35P140 U408 ( .A1(n280), .A2(event_source_y[6]), .B1(
        tap_source_y[6]), .B2(n300), .Z(n137) );
  AO22D0BWP35P140 U409 ( .A1(n280), .A2(event_source_x[1]), .B1(
        tap_source_x[1]), .B2(n282), .Z(n122) );
  AO22D0BWP35P140 U410 ( .A1(n280), .A2(event_source_y[7]), .B1(
        tap_source_y[7]), .B2(n282), .Z(n138) );
  AO22D0BWP35P140 U411 ( .A1(n280), .A2(event_source_x[2]), .B1(
        tap_source_x[2]), .B2(n282), .Z(n123) );
  AO22D0BWP35P140 U412 ( .A1(n280), .A2(event_source_x[3]), .B1(
        tap_source_x[3]), .B2(n282), .Z(n124) );
  AO22D0BWP35P140 U413 ( .A1(n280), .A2(event_source_x[4]), .B1(
        tap_source_x[4]), .B2(n282), .Z(n125) );
  AO22D0BWP35P140 U414 ( .A1(n280), .A2(event_source_x[5]), .B1(
        tap_source_x[5]), .B2(n282), .Z(n126) );
  AO22D0BWP35P140 U415 ( .A1(n280), .A2(event_source_x[7]), .B1(
        tap_source_x[7]), .B2(n282), .Z(n128) );
  AO22D0BWP35P140 U416 ( .A1(n280), .A2(event_source_x[6]), .B1(
        tap_source_x[6]), .B2(n300), .Z(n127) );
  MUX2D0BWP35P140 U417 ( .I0(tap_source_x[6]), .I1(C39_DATA2_7), .S(
        tap_destination_x[0]), .Z(tap_destination_x[7]) );
  AO22D0BWP35P140 U418 ( .A1(n281), .A2(event_tag[6]), .B1(n300), .B2(n358), 
        .Z(n163) );
  AO22D0BWP35P140 U419 ( .A1(n280), .A2(event_tag[8]), .B1(n282), .B2(n391), 
        .Z(n165) );
  AO22D0BWP35P140 U420 ( .A1(n280), .A2(event_tag[10]), .B1(n282), .B2(n389), 
        .Z(n167) );
  AO22D0BWP35P140 U421 ( .A1(n281), .A2(event_tag[5]), .B1(n300), .B2(n437), 
        .Z(n162) );
  AO22D0BWP35P140 U422 ( .A1(n280), .A2(event_time[3]), .B1(n300), .B2(n382), 
        .Z(n156) );
  AO22D0BWP35P140 U423 ( .A1(n280), .A2(event_tag[7]), .B1(n282), .B2(n392), 
        .Z(n164) );
  AO22D0BWP35P140 U424 ( .A1(n280), .A2(event_time[2]), .B1(n300), .B2(n383), 
        .Z(n155) );
  AO22D0BWP35P140 U425 ( .A1(n280), .A2(event_time[1]), .B1(n300), .B2(n384), 
        .Z(n154) );
  AO22D0BWP35P140 U426 ( .A1(n280), .A2(event_tag[1]), .B1(n300), .B2(n396), 
        .Z(n158) );
  AO22D0BWP35P140 U427 ( .A1(n280), .A2(event_tag[0]), .B1(n300), .B2(n397), 
        .Z(n157) );
  AO22D0BWP35P140 U428 ( .A1(n280), .A2(event_tag[11]), .B1(n282), .B2(n388), 
        .Z(n168) );
  AO22D0BWP35P140 U429 ( .A1(n280), .A2(event_tag[3]), .B1(n300), .B2(n394), 
        .Z(n160) );
  AO22D0BWP35P140 U430 ( .A1(n280), .A2(event_tag[9]), .B1(n282), .B2(n390), 
        .Z(n166) );
  AO22D0BWP35P140 U431 ( .A1(n280), .A2(event_tag[2]), .B1(n300), .B2(n395), 
        .Z(n159) );
  AO22D0BWP35P140 U432 ( .A1(n280), .A2(event_time[0]), .B1(n300), .B2(n385), 
        .Z(n153) );
  AO22D0BWP35P140 U433 ( .A1(n280), .A2(event_tag[4]), .B1(n300), .B2(n393), 
        .Z(n161) );
  AO22D0BWP35P140 U434 ( .A1(n280), .A2(event_tag[12]), .B1(n282), .B2(n387), 
        .Z(n169) );
  AO22D0BWP35P140 U435 ( .A1(n280), .A2(event_tag[17]), .B1(n282), .B2(n386), 
        .Z(n174) );
  MUX2D0BWP35P140 U436 ( .I0(tap_source_x[7]), .I1(C39_DATA2_8), .S(
        tap_destination_x[0]), .Z(tap_destination_x[8]) );
  AOI21D0BWP35P140 U437 ( .A1(n300), .A2(n283), .B(n299), .ZN(n290) );
  NR4D0BWP35P140 U438 ( .A1(event_source_x[0]), .A2(event_source_x[3]), .A3(
        event_source_x[5]), .A4(event_source_x[7]), .ZN(n287) );
  ND4D0BWP35P140 U439 ( .A1(n287), .A2(n286), .A3(n285), .A4(n284), .ZN(n288)
         );
  OAI31D0BWP35P140 U440 ( .A1(event_source_x[2]), .A2(event_source_x[1]), .A3(
        n288), .B(n280), .ZN(n296) );
  AOI21D0BWP35P140 U442 ( .A1(n300), .A2(n291), .B(n299), .ZN(n293) );
  OAI21D0BWP35P140 U443 ( .A1(n293), .A2(n334), .B(n296), .ZN(n184) );
  XOR2UD0BWP35P140 U444 ( .A1(N125), .A2(tap_source_x[8]), .Z(n294) );
  XOR2UD0BWP35P140 U445 ( .A1(n294), .A2(DP_OP_70J1_126_5313_n2), .Z(n295) );
  MUX2D0BWP35P140 U446 ( .I0(n295), .I1(tap_source_x[8]), .S(tap_kernel_x[0]), 
        .Z(tap_destination_x[9]) );
  AN2D0BWP35P140 U447 ( .A1(N125), .A2(tap_source_x[0]), .Z(n311) );
  CKND0BWP35P140 U448 ( .I(n296), .ZN(n297) );
  AO22D0BWP35P140 U449 ( .A1(n298), .A2(n297), .B1(n331), .B2(n299), .Z(n192)
         );
  AOI21D0BWP35P140 U450 ( .A1(n300), .A2(pending_q[0]), .B(n299), .ZN(n303) );
  OAI21D0BWP35P140 U451 ( .A1(n303), .A2(n302), .B(n301), .ZN(n183) );
  CKND0BWP35P140 U452 ( .I(n307), .ZN(n308) );
  CKND0BWP35P140 U453 ( .I(tap_kernel_x[1]), .ZN(n304) );
  AOI21D0BWP35P140 U454 ( .A1(n305), .A2(n304), .B(tap_kernel_index[3]), .ZN(
        n306) );
  MUX2ND0BWP35P140 U455 ( .I0(n308), .I1(n307), .S(n306), .ZN(
        tap_kernel_index[1]) );
  DFKCNQD1BWP35P140 fault_q_reg ( .CN(n266), .D(n462), .CP(clk_core), .Q(
        protocol_error) );
  DFKCNQD1BWP35P140 source_channel_q_reg_0_ ( .CN(n141), .D(n327), .CP(
        clk_core), .Q(tap_source_channel[0]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_1_ ( .CN(n327), .D(n142), .CP(
        clk_core), .Q(tap_source_channel[1]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_2_ ( .CN(n327), .D(n143), .CP(
        clk_core), .Q(tap_source_channel[2]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_3_ ( .CN(n327), .D(n144), .CP(
        clk_core), .Q(tap_source_channel[3]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_4_ ( .CN(n327), .D(n145), .CP(
        clk_core), .Q(tap_source_channel[4]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_5_ ( .CN(n327), .D(n146), .CP(
        clk_core), .Q(tap_source_channel[5]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_6_ ( .CN(n327), .D(n147), .CP(
        clk_core), .Q(tap_source_channel[6]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_8_ ( .CN(n327), .D(n149), .CP(
        clk_core), .Q(tap_source_channel[8]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_9_ ( .CN(n327), .D(n150), .CP(
        clk_core), .Q(tap_source_channel[9]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_10_ ( .CN(n327), .D(n151), .CP(
        clk_core), .Q(tap_source_channel[10]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_11_ ( .CN(n327), .D(n152), .CP(
        clk_core), .Q(tap_source_channel[11]) );
  DFKCNQD1BWP35P140 source_y_q_reg_0_ ( .CN(n327), .D(n131), .CP(clk_core), 
        .Q(tap_source_y[0]) );
  DFKCNQD1BWP35P140 source_y_q_reg_2_ ( .CN(n327), .D(n447), .CP(clk_core), 
        .Q(tap_source_y[2]) );
  DFKCNQD1BWP35P140 source_y_q_reg_3_ ( .CN(n327), .D(n444), .CP(clk_core), 
        .Q(tap_source_y[3]) );
  DFKCNQD1BWP35P140 source_y_q_reg_4_ ( .CN(n327), .D(n441), .CP(clk_core), 
        .Q(tap_source_y[4]) );
  DFKCNQD1BWP35P140 source_y_q_reg_5_ ( .CN(n327), .D(n438), .CP(clk_core), 
        .Q(tap_source_y[5]) );
  DFKCNQD1BWP35P140 tag_q_reg_5_ ( .CN(n327), .D(n162), .CP(clk_core), .Q(
        tap_tag[5]) );
  DFKCNQD1BWP35P140 tag_q_reg_13_ ( .CN(n327), .D(n170), .CP(clk_core), .Q(
        tap_tag[13]) );
  DFKCNQD1BWP35P140 tag_q_reg_14_ ( .CN(n327), .D(n171), .CP(clk_core), .Q(
        tap_tag[14]) );
  DFKCNQD1BWP35P140 tag_q_reg_15_ ( .CN(n327), .D(n172), .CP(clk_core), .Q(
        tap_tag[15]) );
  DFKCNQD1BWP35P140 tag_q_reg_16_ ( .CN(n327), .D(n173), .CP(clk_core), .Q(
        tap_tag[16]) );
  DFKCNQD1BWP35P140 tag_q_reg_18_ ( .CN(n327), .D(n175), .CP(clk_core), .Q(
        tap_tag[18]) );
  DFKCNQD1BWP35P140 tag_q_reg_19_ ( .CN(n327), .D(n176), .CP(clk_core), .Q(
        tap_tag[19]) );
  DFKCNQD1BWP35P140 tag_q_reg_20_ ( .CN(n327), .D(n177), .CP(clk_core), .Q(
        tap_tag[20]) );
  DFKCNQD1BWP35P140 tag_q_reg_22_ ( .CN(n327), .D(n179), .CP(clk_core), .Q(
        tap_tag[22]) );
  DFKCNQD1BWP35P140 tag_q_reg_23_ ( .CN(n327), .D(n180), .CP(clk_core), .Q(
        tap_tag[23]) );
  DFKCNQD1BWP35P140 source_x_q_reg_0_ ( .CN(n327), .D(n121), .CP(clk_core), 
        .Q(tap_source_x[0]) );
  DFKCNQD1BWP35P140 source_x_q_reg_1_ ( .CN(n327), .D(n424), .CP(clk_core), 
        .Q(tap_source_x[1]) );
  DFKCNQD1BWP35P140 source_x_q_reg_2_ ( .CN(n327), .D(n421), .CP(clk_core), 
        .Q(tap_source_x[2]) );
  DFKCNQD1BWP35P140 source_x_q_reg_3_ ( .CN(n327), .D(n418), .CP(clk_core), 
        .Q(tap_source_x[3]) );
  DFKCNQD1BWP35P140 source_x_q_reg_4_ ( .CN(n327), .D(n415), .CP(clk_core), 
        .Q(tap_source_x[4]) );
  DFKCNQD1BWP35P140 source_x_q_reg_5_ ( .CN(n327), .D(n412), .CP(clk_core), 
        .Q(tap_source_x[5]) );
  DFKCNQD1BWP35P140 source_x_q_reg_6_ ( .CN(n327), .D(n409), .CP(clk_core), 
        .Q(tap_source_x[6]) );
  DFKCNQD1BWP35P140 source_x_q_reg_7_ ( .CN(n327), .D(n406), .CP(clk_core), 
        .Q(tap_source_x[7]) );
  DFKCNQD1BWP35P140 source_x_q_reg_8_ ( .CN(n327), .D(n129), .CP(clk_core), 
        .Q(tap_source_x[8]) );
  DFKCNQD1BWP35P140 source_y_q_reg_6_ ( .CN(n327), .D(n402), .CP(clk_core), 
        .Q(tap_source_y[6]) );
  DFKCNQD1BWP35P140 source_y_q_reg_7_ ( .CN(n327), .D(n399), .CP(clk_core), 
        .Q(tap_source_y[7]) );
  DFKCNQD1BWP35P140 source_y_q_reg_8_ ( .CN(n327), .D(n139), .CP(clk_core), 
        .Q(tap_source_y[8]) );
  DFKCNQD1BWP35P140 tag_q_reg_0_ ( .CN(n327), .D(n157), .CP(clk_core), .Q(
        tap_tag[0]) );
  DFKCNQD1BWP35P140 tag_q_reg_1_ ( .CN(n327), .D(n158), .CP(clk_core), .Q(
        tap_tag[1]) );
  DFKCNQD1BWP35P140 tag_q_reg_2_ ( .CN(n327), .D(n159), .CP(clk_core), .Q(
        tap_tag[2]) );
  DFKCNQD1BWP35P140 tag_q_reg_3_ ( .CN(n327), .D(n160), .CP(clk_core), .Q(
        tap_tag[3]) );
  DFKCNQD1BWP35P140 tag_q_reg_4_ ( .CN(n327), .D(n161), .CP(clk_core), .Q(
        tap_tag[4]) );
  DFKCNQD1BWP35P140 tag_q_reg_7_ ( .CN(n327), .D(n164), .CP(clk_core), .Q(
        tap_tag[7]) );
  DFKCNQD1BWP35P140 tag_q_reg_8_ ( .CN(n327), .D(n165), .CP(clk_core), .Q(
        tap_tag[8]) );
  DFKCNQD1BWP35P140 tag_q_reg_9_ ( .CN(n327), .D(n166), .CP(clk_core), .Q(
        tap_tag[9]) );
  DFKCNQD1BWP35P140 tag_q_reg_10_ ( .CN(n327), .D(n167), .CP(clk_core), .Q(
        tap_tag[10]) );
  DFKCNQD1BWP35P140 tag_q_reg_11_ ( .CN(n327), .D(n168), .CP(clk_core), .Q(
        tap_tag[11]) );
  DFKCNQD1BWP35P140 tag_q_reg_12_ ( .CN(n327), .D(n169), .CP(clk_core), .Q(
        tap_tag[12]) );
  DFKCNQD1BWP35P140 tag_q_reg_17_ ( .CN(n327), .D(n174), .CP(clk_core), .Q(
        tap_tag[17]) );
  DFKCNQD1BWP35P140 time_q_reg_0_ ( .CN(n327), .D(n153), .CP(clk_core), .Q(
        tap_time[0]) );
  DFKCNQD1BWP35P140 time_q_reg_1_ ( .CN(n327), .D(n154), .CP(clk_core), .Q(
        tap_time[1]) );
  DFKCNQD1BWP35P140 time_q_reg_2_ ( .CN(n327), .D(n155), .CP(clk_core), .Q(
        tap_time[2]) );
  DFKCNQD1BWP35P140 time_q_reg_3_ ( .CN(n327), .D(n156), .CP(clk_core), .Q(
        tap_time[3]) );
  DFKCNQD1BWP35P140 pending_q_reg_8_ ( .CN(n377), .D(n327), .CP(clk_core), .Q(
        pending_q[8]) );
  DFKCNQD1BWP35P140 pending_q_reg_7_ ( .CN(n327), .D(n371), .CP(clk_core), .Q(
        pending_q[7]) );
  DFKCNQD1BWP35P140 pending_q_reg_5_ ( .CN(n327), .D(n367), .CP(clk_core), .Q(
        pending_q[5]) );
  DFKCNQD1BWP35P140 pending_q_reg_3_ ( .CN(n327), .D(n360), .CP(clk_core), .Q(
        pending_q[3]) );
  DFKCNQD1BWP35P140 tag_q_reg_21_ ( .CN(n327), .D(n178), .CP(clk_core), .Q(
        tap_tag[21]) );
  DFKCNQD1BWP35P140 tag_q_reg_6_ ( .CN(n327), .D(n163), .CP(clk_core), .Q(
        tap_tag[6]) );
  DFKCNQD1BWP35P140 stream_last_q_reg ( .CN(n327), .D(n191), .CP(clk_core), 
        .Q(stream_last_q) );
  DFKCNQD1BWP35P140 source_y_q_reg_1_ ( .CN(n327), .D(n354), .CP(clk_core), 
        .Q(tap_source_y[1]) );
  DFKCNQD1BWP35P140 source_channel_q_reg_7_ ( .CN(n327), .D(n148), .CP(
        clk_core), .Q(tap_source_channel[7]) );
  DFKCNQD1BWP35P140 pending_q_reg_4_ ( .CN(n327), .D(n347), .CP(clk_core), .Q(
        pending_q[4]) );
  DFKCNQD1BWP35P140 pending_q_reg_1_ ( .CN(n327), .D(n341), .CP(clk_core), .Q(
        pending_q[1]) );
  DFKCNQD1BWP35P140 pending_q_reg_6_ ( .CN(n327), .D(n337), .CP(clk_core), .Q(
        pending_q[6]) );
  DFKCNQD1BWP35P140 pending_q_reg_2_ ( .CN(n327), .D(n332), .CP(clk_core), .Q(
        pending_q[2]) );
  DFKCNQD1BWP35P140 pending_q_reg_0_ ( .CN(n327), .D(n192), .CP(clk_core), .Q(
        pending_q[0]) );
  NR2D0BWP35P140 U246 ( .A1(event_input_width[8]), .A2(event_input_width[6]), 
        .ZN(n234) );
  ND2D0BWP35P140 U254 ( .A1(n325), .A2(n271), .ZN(n248) );
  ND3D0BWP35P140 U267 ( .A1(event_input_width[9]), .A2(n234), .A3(n233), .ZN(
        n235) );
  NR2D0BWP35P140 U268 ( .A1(pending_q[4]), .A2(n248), .ZN(n247) );
  ND3D0BWP35P140 U269 ( .A1(pending_q[7]), .A2(n247), .A3(n322), .ZN(n265) );
  NR2D0BWP35P140 U274 ( .A1(n307), .A2(n198), .ZN(n199) );
  AN2D0BWP35P140 U275 ( .A1(tap_valid), .A2(tap_ready), .Z(tap_accept) );
  NR2D0BWP35P140 U276 ( .A1(tap_kernel_x[0]), .A2(n241), .ZN(tap_kernel_x[1])
         );
  NR3D0BWP35P140 U277 ( .A1(n253), .A2(protocol_error), .A3(n254), .ZN(
        event_ready) );
  ND2D0BWP35P140 U319 ( .A1(n338), .A2(n247), .ZN(n283) );
  CKND0BWP35P140 U321 ( .I(n338), .ZN(n322) );
  CKND0BWP35P140 U323 ( .I(pending_q[4]), .ZN(n323) );
  CKND0BWP35P140 U340 ( .I(pending_q[3]), .ZN(n324) );
  CKND0BWP35P140 U344 ( .I(n366), .ZN(n325) );
  OAI21D0BWP35P140 U349 ( .A1(n268), .A2(n374), .B(n276), .ZN(n189) );
  CKND0BWP35P140 U356 ( .I(pending_q[7]), .ZN(n326) );
  CKND0BWP35P140 U357 ( .I(n463), .ZN(n328) );
  DFKCSND1BWP35P140 busy_q_reg ( .D(n254), .SN(n255), .CN(n266), .CP(clk_core), 
        .Q(busy), .QN(n252) );
  TIEHBWP35P140 U279 ( .Z(n327) );
  INVD1BWP35P140 U287 ( .I(n327), .ZN(tap_source_x[9]) );
  INVD1BWP35P140 U316 ( .I(n327), .ZN(tap_source_y[9]) );
  DEL075MD1BWP35P140 U338 ( .I(pending_q[0]), .Z(n331) );
  CKBD1BWP35P140 U341 ( .I(n333), .Z(n332) );
  CKBD1BWP35P140 U441 ( .I(n335), .Z(n333) );
  CKBD1BWP35P140 U456 ( .I(n336), .Z(n334) );
  CKBD1BWP35P140 U457 ( .I(n184), .Z(n335) );
  CKBD1BWP35P140 U458 ( .I(n292), .Z(n336) );
  OAI21D0BWP35P140 U459 ( .A1(n290), .A2(n322), .B(n296), .ZN(n188) );
  CKBD1BWP35P140 U460 ( .I(n339), .Z(n337) );
  DEL050MD1BWP35P140 U461 ( .I(pending_q[6]), .Z(n338) );
  CKBD1BWP35P140 U462 ( .I(n340), .Z(n339) );
  CKBD1BWP35P140 U463 ( .I(n188), .Z(n340) );
  CKBD1BWP35P140 U464 ( .I(n342), .Z(n341) );
  CKBD1BWP35P140 U465 ( .I(n343), .Z(n342) );
  CKBD1BWP35P140 U466 ( .I(n344), .Z(n343) );
  CKBD1BWP35P140 U467 ( .I(n345), .Z(n344) );
  CKBD1BWP35P140 U468 ( .I(n346), .Z(n345) );
  CKBD1BWP35P140 U469 ( .I(n183), .Z(n346) );
  CKBD1BWP35P140 U470 ( .I(n348), .Z(n347) );
  CKBD1BWP35P140 U471 ( .I(n349), .Z(n348) );
  CKBD1BWP35P140 U472 ( .I(n351), .Z(n349) );
  CKBD1BWP35P140 U473 ( .I(n352), .Z(n350) );
  CKBD1BWP35P140 U474 ( .I(n186), .Z(n351) );
  CKBD1BWP35P140 U475 ( .I(n323), .Z(n352) );
  DEL050MD1BWP35P140 U476 ( .I(tap_source_channel[7]), .Z(n353) );
  CKBD1BWP35P140 U477 ( .I(n355), .Z(n354) );
  CKBD1BWP35P140 U478 ( .I(n356), .Z(n355) );
  CKBD1BWP35P140 U479 ( .I(n132), .Z(n356) );
  DEL075MD1BWP35P140 U480 ( .I(stream_last_q), .Z(n357) );
  DEL050MD1BWP35P140 U481 ( .I(tap_tag[6]), .Z(n358) );
  DEL050MD1BWP35P140 U482 ( .I(tap_tag[21]), .Z(n359) );
  CKBD1BWP35P140 U483 ( .I(n361), .Z(n360) );
  CKBD1BWP35P140 U484 ( .I(n362), .Z(n361) );
  CKBD1BWP35P140 U485 ( .I(n364), .Z(n362) );
  CKBD1BWP35P140 U486 ( .I(n365), .Z(n363) );
  CKBD1BWP35P140 U487 ( .I(n185), .Z(n364) );
  CKBD1BWP35P140 U488 ( .I(n324), .Z(n365) );
  BUFFD0BWP35P140 U489 ( .I(pending_q[5]), .Z(n366) );
  CKBD1BWP35P140 U490 ( .I(n368), .Z(n367) );
  CKBD1BWP35P140 U491 ( .I(n369), .Z(n368) );
  CKBD1BWP35P140 U492 ( .I(n370), .Z(n369) );
  CKBD1BWP35P140 U493 ( .I(n187), .Z(n370) );
  CKBD1BWP35P140 U494 ( .I(n372), .Z(n371) );
  CKBD1BWP35P140 U495 ( .I(n373), .Z(n372) );
  CKBD1BWP35P140 U496 ( .I(n375), .Z(n373) );
  CKBD1BWP35P140 U497 ( .I(n376), .Z(n374) );
  CKBD1BWP35P140 U498 ( .I(n189), .Z(n375) );
  CKBD1BWP35P140 U499 ( .I(n326), .Z(n376) );
  CKBD1BWP35P140 U500 ( .I(n380), .Z(n377) );
  CKBD1BWP35P140 U501 ( .I(n379), .Z(n378) );
  CKBD1BWP35P140 U502 ( .I(n381), .Z(n379) );
  CKBD1BWP35P140 U503 ( .I(n190), .Z(n380) );
  CKBD1BWP35P140 U504 ( .I(n260), .Z(n381) );
  DEL050MD1BWP35P140 U505 ( .I(tap_time[3]), .Z(n382) );
  DEL050MD1BWP35P140 U506 ( .I(tap_time[2]), .Z(n383) );
  DEL050MD1BWP35P140 U507 ( .I(tap_time[1]), .Z(n384) );
  DEL050MD1BWP35P140 U508 ( .I(tap_time[0]), .Z(n385) );
  DEL050MD1BWP35P140 U509 ( .I(tap_tag[17]), .Z(n386) );
  DEL050MD1BWP35P140 U510 ( .I(tap_tag[12]), .Z(n387) );
  DEL050MD1BWP35P140 U511 ( .I(tap_tag[11]), .Z(n388) );
  DEL050MD1BWP35P140 U512 ( .I(tap_tag[10]), .Z(n389) );
  DEL050MD1BWP35P140 U513 ( .I(tap_tag[9]), .Z(n390) );
  DEL050MD1BWP35P140 U514 ( .I(tap_tag[8]), .Z(n391) );
  DEL050MD1BWP35P140 U515 ( .I(tap_tag[7]), .Z(n392) );
  DEL050MD1BWP35P140 U516 ( .I(tap_tag[4]), .Z(n393) );
  DEL050MD1BWP35P140 U517 ( .I(tap_tag[3]), .Z(n394) );
  DEL050MD1BWP35P140 U518 ( .I(tap_tag[2]), .Z(n395) );
  DEL050MD1BWP35P140 U519 ( .I(tap_tag[1]), .Z(n396) );
  DEL050MD1BWP35P140 U520 ( .I(tap_tag[0]), .Z(n397) );
  DEL050MD1BWP35P140 U521 ( .I(tap_source_y[8]), .Z(n398) );
  CKBD1BWP35P140 U522 ( .I(n400), .Z(n399) );
  CKBD1BWP35P140 U523 ( .I(n401), .Z(n400) );
  CKBD1BWP35P140 U524 ( .I(n138), .Z(n401) );
  CKBD1BWP35P140 U525 ( .I(n403), .Z(n402) );
  CKBD1BWP35P140 U526 ( .I(n404), .Z(n403) );
  CKBD1BWP35P140 U527 ( .I(n137), .Z(n404) );
  DEL050MD1BWP35P140 U528 ( .I(tap_source_x[8]), .Z(n405) );
  CKBD1BWP35P140 U529 ( .I(n407), .Z(n406) );
  CKBD1BWP35P140 U530 ( .I(n408), .Z(n407) );
  CKBD1BWP35P140 U531 ( .I(n128), .Z(n408) );
  CKBD1BWP35P140 U532 ( .I(n410), .Z(n409) );
  CKBD1BWP35P140 U533 ( .I(n411), .Z(n410) );
  CKBD1BWP35P140 U534 ( .I(n127), .Z(n411) );
  CKBD1BWP35P140 U535 ( .I(n413), .Z(n412) );
  CKBD1BWP35P140 U536 ( .I(n414), .Z(n413) );
  CKBD1BWP35P140 U537 ( .I(n126), .Z(n414) );
  CKBD1BWP35P140 U538 ( .I(n416), .Z(n415) );
  CKBD1BWP35P140 U539 ( .I(n417), .Z(n416) );
  CKBD1BWP35P140 U540 ( .I(n125), .Z(n417) );
  CKBD1BWP35P140 U541 ( .I(n419), .Z(n418) );
  CKBD1BWP35P140 U542 ( .I(n420), .Z(n419) );
  CKBD1BWP35P140 U543 ( .I(n124), .Z(n420) );
  CKBD1BWP35P140 U544 ( .I(n422), .Z(n421) );
  CKBD1BWP35P140 U545 ( .I(n423), .Z(n422) );
  CKBD1BWP35P140 U546 ( .I(n123), .Z(n423) );
  CKBD1BWP35P140 U547 ( .I(n425), .Z(n424) );
  CKBD1BWP35P140 U548 ( .I(n426), .Z(n425) );
  CKBD1BWP35P140 U549 ( .I(n122), .Z(n426) );
  DEL050MD1BWP35P140 U550 ( .I(tap_source_x[0]), .Z(n427) );
  DEL050MD1BWP35P140 U551 ( .I(tap_tag[23]), .Z(n428) );
  DEL050MD1BWP35P140 U552 ( .I(tap_tag[22]), .Z(n429) );
  DEL050MD1BWP35P140 U553 ( .I(tap_tag[20]), .Z(n430) );
  DEL050MD1BWP35P140 U554 ( .I(tap_tag[19]), .Z(n431) );
  DEL050MD1BWP35P140 U555 ( .I(tap_tag[18]), .Z(n432) );
  DEL050MD1BWP35P140 U556 ( .I(tap_tag[16]), .Z(n433) );
  DEL050MD1BWP35P140 U557 ( .I(tap_tag[15]), .Z(n434) );
  DEL050MD1BWP35P140 U558 ( .I(tap_tag[14]), .Z(n435) );
  DEL050MD1BWP35P140 U559 ( .I(tap_tag[13]), .Z(n436) );
  DEL050MD1BWP35P140 U560 ( .I(tap_tag[5]), .Z(n437) );
  CKBD1BWP35P140 U561 ( .I(n439), .Z(n438) );
  CKBD1BWP35P140 U562 ( .I(n440), .Z(n439) );
  CKBD1BWP35P140 U563 ( .I(n136), .Z(n440) );
  CKBD1BWP35P140 U564 ( .I(n442), .Z(n441) );
  CKBD1BWP35P140 U565 ( .I(n443), .Z(n442) );
  CKBD1BWP35P140 U566 ( .I(n135), .Z(n443) );
  CKBD1BWP35P140 U567 ( .I(n445), .Z(n444) );
  CKBD1BWP35P140 U568 ( .I(n446), .Z(n445) );
  CKBD1BWP35P140 U569 ( .I(n134), .Z(n446) );
  CKBD1BWP35P140 U570 ( .I(n448), .Z(n447) );
  CKBD1BWP35P140 U571 ( .I(n449), .Z(n448) );
  CKBD1BWP35P140 U572 ( .I(n133), .Z(n449) );
  DEL050MD1BWP35P140 U573 ( .I(tap_source_y[0]), .Z(n450) );
  DEL050MD1BWP35P140 U574 ( .I(tap_source_channel[11]), .Z(n451) );
  DEL050MD1BWP35P140 U575 ( .I(tap_source_channel[10]), .Z(n452) );
  DEL050MD1BWP35P140 U576 ( .I(tap_source_channel[9]), .Z(n453) );
  DEL050MD1BWP35P140 U577 ( .I(tap_source_channel[8]), .Z(n454) );
  DEL050MD1BWP35P140 U578 ( .I(tap_source_channel[6]), .Z(n455) );
  DEL050MD1BWP35P140 U579 ( .I(tap_source_channel[5]), .Z(n456) );
  DEL050MD1BWP35P140 U580 ( .I(tap_source_channel[4]), .Z(n457) );
  DEL050MD1BWP35P140 U581 ( .I(tap_source_channel[3]), .Z(n458) );
  DEL050MD1BWP35P140 U582 ( .I(tap_source_channel[2]), .Z(n459) );
  DEL050MD1BWP35P140 U583 ( .I(tap_source_channel[1]), .Z(n460) );
  DEL050MD1BWP35P140 U584 ( .I(tap_source_channel[0]), .Z(n461) );
  CKBD1BWP35P140 U585 ( .I(n464), .Z(n462) );
  CKBD1BWP35P140 U586 ( .I(n240), .Z(n463) );
  CKBD1BWP35P140 U587 ( .I(n328), .Z(n464) );
  DEL075MD1BWP35P140 U588 ( .I(n252), .Z(n465) );
endmodule

