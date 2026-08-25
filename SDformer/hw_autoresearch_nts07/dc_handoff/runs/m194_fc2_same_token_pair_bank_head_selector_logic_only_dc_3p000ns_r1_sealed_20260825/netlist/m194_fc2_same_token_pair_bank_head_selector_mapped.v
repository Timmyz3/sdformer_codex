/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : V-2023.12-SP3
// Date      : Tue Aug 25 02:31:15 2026
/////////////////////////////////////////////////////////////


module m194_fc2_same_token_pair_bank_head_selector ( clk_core, rst_core, 
        pair_valid, pair_ready, window_valid, window_token_tag, 
        window_bank_count, window_head_channel, pair_accept, issue_valid, 
        issue_ready, issue_token_tag, issue_source_count, issue_bank_valid, 
        issue_selected_window, issue_source_channel, issue_pair_last, 
        issue_accept, protocol_error, busy );
  input [1:0] window_valid;
  input [47:0] window_token_tag;
  input [127:0] window_bank_count;
  input [191:0] window_head_channel;
  output [23:0] issue_token_tag;
  output [3:0] issue_source_count;
  output [7:0] issue_bank_valid;
  output [7:0] issue_selected_window;
  output [95:0] issue_source_channel;
  input clk_core, rst_core, pair_valid, issue_ready;
  output pair_ready, pair_accept, issue_valid, issue_pair_last, issue_accept,
         protocol_error, busy;
  wire   fault_q, n310, n311, n312, n313, n314, n315, n316, n317, n318, n319,
         n320, n321, n322, n323, n324, n325, n326, n327, n328, n329, n330,
         n331, n332, n333, n334, n335, n336, n337, n338, n339, n340, n341,
         n342, n343, n344, n345, n346, n347, n348, n349, n350, n351, n352,
         n353, n355, n356, n357, n358, n359, n360, n361, n362, n363, n367,
         n368, n369, n370, n371, n372, n373, n374, n375, n378, n379, n380,
         n381, n382, n383, n384, n385, n386, n387, n389, n391, n392, n393,
         n394, n395, n396, n397, n398, n399, n401, n402, n403, n404, n405,
         n406, n407, n408, n409, n410, n411, n412, n415, n416, n417, n418,
         n419, n420, n421, n422, n423, n424, n426, n427, n428, n429, n430,
         n431, n432, n433, n434, n435, n436, n437, n439, n440, n441, n442,
         n443, n444, n445, n446, n447, n448, n449, n450, n451, n452, n466,
         n467, n468, n469, n470, n471, n472, n473, n474, n475, n476, n477,
         n478, n479, n480, n481, n482, n483, n484, n485, n486, n487, n488,
         n489, n490, n491, n492, n493, n494, n495, n496, n497, n498, n499,
         n500, n501, n502, n503, n504, n505, n506, n507, n508, n509, n510,
         n511, n512, n513, n514, n515, n516, n517, n518, n519, n520, n521,
         n522, n523, n524, n525, n526, n527, n528, n529, n530, n531, n532,
         n533, n534, n535, n536, n537, n538, n539, n540, n541, n542, n543,
         n544, n545, n546, n547, n548, n549, n550, n551, n552, n553, n554,
         n555, n556, n557, n558, n559, n560, n561, n562, n563, n564, n565,
         n566, n567, n568, n569, n570, n571, n572, n573, n574, n575, n576,
         n577, n578, n579, n580, n581, n582, n583, n584, n585, n586, n587,
         n588, n589, n590, n591, n592, n593, n594, n595, n596, n597, n598,
         n599, n600, n601, n602, n603, n604, n605, n606, n607, n608, n609,
         n610, n611, n612, n613, n614, n615, n616, n617, n618, n619, n620,
         n621, n622, n623, n624, n625, n626, n627, n628, n629, n630, n631,
         n632, n633, n634, n635, n636, n637, n638, n639, n640, n641, n642,
         n643, n644, n645, n646, n647, n648, n649, n650, n651, n652, n653,
         n654, n655, n656, n657, n658, n659, n660, n661, n662, n663, n664,
         n665, n666, n667, n668, n669, n670, n671, n672, n673, n674, n675,
         n676, n677, n678, n679, n680, n681, n682, n683, n684, n685, n686,
         n687, n688, n689, n690, n691, n692, n693, n694, n695, n696, n697,
         n698, n699, n700, n701, n702, n703, n704, n705, n706, n707, n708,
         n709, n710, n711, n712, n713, n714, n715, n716, n717, n718, n719,
         n720, n721, n722, n723, n724, n725, n726, n727, n728, n729, n730,
         n731, n732, n733, n734, n735, n736, n737, n738, n739, n740, n741,
         n742, n743, n744, n745, n746, n747, n748, n749, n750, n751, n752,
         n753, n754, n755, n756, n757, n758, n759, n760, n761, n762, n763,
         n764, n765, n766, n767, n768, n769, n770, n771, n772, n773, n774,
         n775, n776, n777, n778, n779, n780, n781, n782, n783, n784, n785,
         n786, n787, n788, n789, n790, n791, n792, n793, n794, n795, n796,
         n797, n798, n799, n800, n801, n802, n803, n804, n805, n806, n807,
         n808, n809, n810, n811, n812, n813, n814, n815, n816, n817, n818,
         n819, n820, n821, n822, n823, n824, n825, n826, n828, n841;

  CKND0BWP35P140 U583 ( .I(n767), .ZN(n812) );
  OAI211D0BWP35P140 U584 ( .A1(n731), .A2(n575), .B(n574), .C(n573), .ZN(n592)
         );
  OAI211D0BWP35P140 U585 ( .A1(n729), .A2(n590), .B(n589), .C(n588), .ZN(n591)
         );
  AOI32D0BWP35P140 U586 ( .A1(n571), .A2(n674), .A3(n570), .B1(window_valid[0]), .B2(n569), .ZN(n593) );
  AOI211D0BWP35P140 U587 ( .A1(n667), .A2(n521), .B(n520), .C(n519), .ZN(n596)
         );
  OAI21D0BWP35P140 U588 ( .A1(window_head_channel[108]), .A2(n587), .B(n666), 
        .ZN(n588) );
  OAI211D0BWP35P140 U589 ( .A1(n654), .A2(n518), .B(n517), .C(n516), .ZN(n519)
         );
  AOI211D0BWP35P140 U590 ( .A1(n615), .A2(n586), .B(n585), .C(n584), .ZN(n589)
         );
  CKND2D1BWP35P140 U591 ( .A1(n497), .A2(n788), .ZN(n477) );
  CKND2D1BWP35P140 U592 ( .A1(n789), .A2(n777), .ZN(n470) );
  AOI31D0BWP35P140 U593 ( .A1(window_head_channel[0]), .A2(
        window_head_channel[2]), .A3(window_head_channel[1]), .B(n641), .ZN(
        n585) );
  OAI211D0BWP35P140 U594 ( .A1(n656), .A2(n583), .B(n582), .C(n581), .ZN(n584)
         );
  AOI31D0BWP35P140 U595 ( .A1(window_head_channel[96]), .A2(
        window_head_channel[98]), .A3(window_head_channel[97]), .B(n643), .ZN(
        n594) );
  AOI22D0BWP35P140 U596 ( .A1(n616), .A2(n579), .B1(n733), .B2(n578), .ZN(n582) );
  CKND2D1BWP35P140 U597 ( .A1(n488), .A2(n487), .ZN(n799) );
  OAI21D0BWP35P140 U598 ( .A1(window_head_channel[146]), .A2(n572), .B(n734), 
        .ZN(n574) );
  CKND2D1BWP35P140 U599 ( .A1(n468), .A2(n467), .ZN(n533) );
  AOI211D0BWP35P140 U601 ( .A1(window_bank_count[24]), .A2(
        window_bank_count[88]), .B(n529), .C(n486), .ZN(n488) );
  OAI211D0BWP35P140 U602 ( .A1(n481), .A2(n480), .B(n502), .C(n538), .ZN(n482)
         );
  CKND2D1BWP35P140 U603 ( .A1(n504), .A2(n503), .ZN(n577) );
  CKND2D1BWP35P140 U604 ( .A1(n506), .A2(n505), .ZN(n615) );
  AO211D0BWP35P140 U606 ( .A1(window_bank_count[104]), .A2(
        window_bank_count[40]), .B(window_bank_count[43]), .C(
        window_bank_count[41]), .Z(n484) );
  CKND0BWP35P140 U607 ( .I(window_valid[0]), .ZN(n674) );
  AOI211D0BWP35P140 U609 ( .A1(window_bank_count[48]), .A2(
        window_bank_count[112]), .B(window_bank_count[113]), .C(
        window_bank_count[114]), .ZN(n466) );
  AOI211D0BWP35P140 U610 ( .A1(window_bank_count[80]), .A2(
        window_bank_count[16]), .B(window_bank_count[81]), .C(
        window_bank_count[17]), .ZN(n783) );
  AO211D0BWP35P140 U611 ( .A1(window_bank_count[8]), .A2(window_bank_count[72]), .B(window_bank_count[14]), .C(window_bank_count[15]), .Z(n469) );
  AO211D0BWP35P140 U613 ( .A1(window_bank_count[64]), .A2(window_bank_count[0]), .B(window_bank_count[65]), .C(window_bank_count[7]), .Z(n472) );
  AOI211D0BWP35P140 U614 ( .A1(window_bank_count[56]), .A2(
        window_bank_count[120]), .B(window_bank_count[62]), .C(
        window_bank_count[63]), .ZN(n475) );
  NR2D0BWP35P140 U628 ( .A1(rst_core), .A2(pair_accept), .ZN(n776) );
  CKND0BWP35P140 U629 ( .I(rst_core), .ZN(n828) );
  DEL025D1BWP35P140 U630 ( .I(issue_valid), .Z(busy) );
  AN2D0BWP35P140 U632 ( .A1(issue_ready), .A2(issue_valid), .Z(issue_accept)
         );
  NR2D1BWP35P140 U635 ( .A1(window_bank_count[116]), .A2(
        window_bank_count[115]), .ZN(n541) );
  NR2D1BWP35P140 U636 ( .A1(window_bank_count[53]), .A2(window_bank_count[52]), 
        .ZN(n510) );
  ND4D0BWP35P140 U637 ( .A1(n509), .A2(n541), .A3(n510), .A4(n466), .ZN(n786)
         );
  OR2D1BWP35P140 U638 ( .A1(window_bank_count[55]), .A2(window_bank_count[54]), 
        .Z(n507) );
  OR4D1BWP35P140 U639 ( .A1(window_bank_count[118]), .A2(
        window_bank_count[119]), .A3(window_bank_count[117]), .A4(n507), .Z(
        n796) );
  NR4D0BWP35P140 U640 ( .A1(window_bank_count[112]), .A2(n508), .A3(n786), 
        .A4(n796), .ZN(n810) );
  NR4D0BWP35P140 U642 ( .A1(window_bank_count[78]), .A2(window_bank_count[74]), 
        .A3(window_bank_count[76]), .A4(window_bank_count[75]), .ZN(n467) );
  NR4D0BWP35P140 U643 ( .A1(window_bank_count[12]), .A2(window_bank_count[9]), 
        .A3(n469), .A4(n533), .ZN(n789) );
  NR2D1BWP35P140 U644 ( .A1(window_bank_count[13]), .A2(window_bank_count[11]), 
        .ZN(n777) );
  NR4D0BWP35P140 U645 ( .A1(window_bank_count[8]), .A2(window_bank_count[72]), 
        .A3(window_bank_count[10]), .A4(n470), .ZN(n806) );
  NR2D1BWP35P140 U646 ( .A1(window_bank_count[2]), .A2(window_bank_count[0]), 
        .ZN(n473) );
  OR4D1BWP35P140 U647 ( .A1(window_bank_count[69]), .A2(window_bank_count[67]), 
        .A3(window_bank_count[6]), .A4(window_bank_count[4]), .Z(n471) );
  NR4D0BWP35P140 U648 ( .A1(window_bank_count[68]), .A2(window_bank_count[66]), 
        .A3(window_bank_count[3]), .A4(n471), .ZN(n793) );
  NR4D0BWP35P140 U649 ( .A1(window_bank_count[71]), .A2(window_bank_count[70]), 
        .A3(window_bank_count[1]), .A4(n472), .ZN(n792) );
  NR2D1BWP35P140 U652 ( .A1(window_bank_count[58]), .A2(window_bank_count[56]), 
        .ZN(n497) );
  NR2D1BWP35P140 U653 ( .A1(window_bank_count[122]), .A2(
        window_bank_count[121]), .ZN(n534) );
  NR4D0BWP35P140 U654 ( .A1(window_bank_count[124]), .A2(
        window_bank_count[123]), .A3(window_bank_count[127]), .A4(
        window_bank_count[125]), .ZN(n536) );
  IND4D1BWP35P140 U655 ( .A1(window_bank_count[57]), .B1(n534), .B2(n536), 
        .B3(n475), .ZN(n476) );
  NR4D0BWP35P140 U657 ( .A1(window_bank_count[61]), .A2(window_bank_count[120]), .A3(window_bank_count[59]), .A4(n477), .ZN(n808) );
  ND4D0BWP35P140 U658 ( .A1(n810), .A2(n806), .A3(n807), .A4(n808), .ZN(n598)
         );
  NR2D1BWP35P140 U659 ( .A1(window_bank_count[84]), .A2(window_bank_count[82]), 
        .ZN(n532) );
  INR4D0BWP35P140 U660 ( .A1(n532), .B1(window_bank_count[83]), .B2(
        window_bank_count[20]), .B3(window_bank_count[19]), .ZN(n795) );
  NR4D0BWP35P140 U662 ( .A1(window_bank_count[86]), .A2(window_bank_count[87]), 
        .A3(window_bank_count[85]), .A4(n478), .ZN(n794) );
  NR4D0BWP35P140 U663 ( .A1(window_bank_count[80]), .A2(window_bank_count[16]), 
        .A3(window_bank_count[21]), .A4(window_bank_count[18]), .ZN(n479) );
  ND4D0BWP35P140 U664 ( .A1(n795), .A2(n794), .A3(n783), .A4(n479), .ZN(n614)
         );
  CKND0BWP35P140 U665 ( .I(window_bank_count[32]), .ZN(n481) );
  CKND0BWP35P140 U666 ( .I(window_bank_count[96]), .ZN(n480) );
  NR2D1BWP35P140 U667 ( .A1(window_bank_count[39]), .A2(window_bank_count[38]), 
        .ZN(n502) );
  NR4D0BWP35P140 U668 ( .A1(window_bank_count[101]), .A2(window_bank_count[98]), .A3(window_bank_count[100]), .A4(window_bank_count[99]), .ZN(n538) );
  NR4D0BWP35P140 U669 ( .A1(window_bank_count[103]), .A2(
        window_bank_count[102]), .A3(window_bank_count[36]), .A4(n482), .ZN(
        n803) );
  NR3D0P7BWP35P140 U670 ( .A1(window_bank_count[37]), .A2(
        window_bank_count[35]), .A3(window_bank_count[34]), .ZN(n782) );
  NR4D0BWP35P140 U671 ( .A1(window_bank_count[96]), .A2(window_bank_count[32]), 
        .A3(window_bank_count[97]), .A4(window_bank_count[33]), .ZN(n483) );
  ND3D1BWP35P140 U672 ( .A1(n803), .A2(n782), .A3(n483), .ZN(n613) );
  NR4D0BWP35P140 U674 ( .A1(window_bank_count[105]), .A2(
        window_bank_count[107]), .A3(window_bank_count[106]), .A4(n484), .ZN(
        n790) );
  NR4D0BWP35P140 U675 ( .A1(window_bank_count[111]), .A2(
        window_bank_count[110]), .A3(window_bank_count[47]), .A4(
        window_bank_count[46]), .ZN(n791) );
  NR2D1BWP35P140 U676 ( .A1(window_bank_count[40]), .A2(window_bank_count[42]), 
        .ZN(n493) );
  NR2D1BWP35P140 U677 ( .A1(window_bank_count[108]), .A2(window_bank_count[44]), .ZN(n779) );
  ND4D0BWP35P140 U678 ( .A1(n790), .A2(n791), .A3(n493), .A4(n779), .ZN(n485)
         );
  NR4D0BWP35P140 U679 ( .A1(window_bank_count[104]), .A2(
        window_bank_count[109]), .A3(window_bank_count[45]), .A4(n485), .ZN(
        n809) );
  NR2D1BWP35P140 U680 ( .A1(window_bank_count[27]), .A2(window_bank_count[26]), 
        .ZN(n778) );
  IND2D1BWP35P140 U681 ( .A1(window_bank_count[24]), .B1(n778), .ZN(n489) );
  OR2D1BWP35P140 U682 ( .A1(window_bank_count[92]), .A2(window_bank_count[91]), 
        .Z(n529) );
  OR4D1BWP35P140 U683 ( .A1(window_bank_count[89]), .A2(window_bank_count[90]), 
        .A3(window_bank_count[25]), .A4(window_bank_count[28]), .Z(n486) );
  OR2D1BWP35P140 U684 ( .A1(window_bank_count[31]), .A2(window_bank_count[30]), 
        .Z(n499) );
  NR4D0BWP35P140 U685 ( .A1(window_bank_count[94]), .A2(window_bank_count[95]), 
        .A3(window_bank_count[93]), .A4(n499), .ZN(n487) );
  NR4D0BWP35P140 U686 ( .A1(window_bank_count[88]), .A2(window_bank_count[29]), 
        .A3(n489), .A4(n799), .ZN(n813) );
  NR4D0BWP35P140 U688 ( .A1(window_bank_count[8]), .A2(window_bank_count[9]), 
        .A3(window_bank_count[11]), .A4(window_bank_count[10]), .ZN(n491) );
  NR4D0BWP35P140 U689 ( .A1(window_bank_count[12]), .A2(window_bank_count[15]), 
        .A3(window_bank_count[14]), .A4(window_bank_count[13]), .ZN(n490) );
  IND3D1BWP35P140 U690 ( .A1(window_head_channel[12]), .B1(
        window_head_channel[13]), .B2(window_head_channel[14]), .ZN(n521) );
  NR4D0BWP35P140 U692 ( .A1(window_bank_count[47]), .A2(window_bank_count[46]), 
        .A3(window_bank_count[45]), .A4(window_bank_count[44]), .ZN(n492) );
  ND3D1BWP35P140 U693 ( .A1(n494), .A2(n493), .A3(n492), .ZN(n576) );
  NR2D1BWP35P140 U694 ( .A1(window_bank_count[63]), .A2(window_bank_count[62]), 
        .ZN(n496) );
  NR4D0BWP35P140 U695 ( .A1(window_bank_count[61]), .A2(window_bank_count[60]), 
        .A3(window_bank_count[57]), .A4(window_bank_count[59]), .ZN(n495) );
  ND3D1BWP35P140 U696 ( .A1(n497), .A2(n496), .A3(n495), .ZN(n624) );
  NR4D0BWP35P140 U697 ( .A1(window_bank_count[24]), .A2(window_bank_count[25]), 
        .A3(window_bank_count[28]), .A4(window_bank_count[29]), .ZN(n498) );
  IND3D1BWP35P140 U698 ( .A1(n499), .B1(n778), .B2(n498), .ZN(n514) );
  NR4D0BWP35P140 U700 ( .A1(window_bank_count[32]), .A2(window_bank_count[36]), 
        .A3(window_bank_count[33]), .A4(window_bank_count[37]), .ZN(n500) );
  ND3D1BWP35P140 U701 ( .A1(n502), .A2(n501), .A3(n500), .ZN(n733) );
  NR4D0BWP35P140 U702 ( .A1(n576), .A2(n624), .A3(n514), .A4(n733), .ZN(n513)
         );
  NR4D0BWP35P140 U703 ( .A1(window_bank_count[5]), .A2(window_bank_count[4]), 
        .A3(window_bank_count[3]), .A4(window_bank_count[1]), .ZN(n504) );
  NR4D0BWP35P140 U704 ( .A1(window_bank_count[2]), .A2(window_bank_count[0]), 
        .A3(window_bank_count[7]), .A4(window_bank_count[6]), .ZN(n503) );
  NR4D0BWP35P140 U705 ( .A1(window_bank_count[19]), .A2(window_bank_count[17]), 
        .A3(window_bank_count[16]), .A4(window_bank_count[18]), .ZN(n506) );
  NR4D0BWP35P140 U706 ( .A1(window_bank_count[20]), .A2(window_bank_count[23]), 
        .A3(window_bank_count[22]), .A4(window_bank_count[21]), .ZN(n505) );
  NR2D1BWP35P140 U707 ( .A1(n508), .A2(n507), .ZN(n511) );
  ND3D1BWP35P140 U708 ( .A1(n511), .A2(n510), .A3(n509), .ZN(n630) );
  NR4D0BWP35P140 U709 ( .A1(n577), .A2(n615), .A3(n630), .A4(n667), .ZN(n512)
         );
  AOI21D0BWP35P140 U710 ( .A1(n513), .A2(n512), .B(window_valid[1]), .ZN(n520)
         );
  CKND0BWP35P140 U711 ( .I(n514), .ZN(n654) );
  INR3D0BWP35P140 U712 ( .A1(window_head_channel[38]), .B1(
        window_head_channel[37]), .B2(window_head_channel[36]), .ZN(n518) );
  OAI31D0BWP35P140 U713 ( .A1(window_head_channel[85]), .A2(
        window_head_channel[84]), .A3(window_head_channel[86]), .B(n624), .ZN(
        n517) );
  CKND0BWP35P140 U714 ( .I(window_head_channel[72]), .ZN(n515) );
  OAI31D0BWP35P140 U715 ( .A1(window_head_channel[74]), .A2(
        window_head_channel[73]), .A3(n515), .B(n630), .ZN(n516) );
  OR4D1BWP35P140 U717 ( .A1(window_bank_count[65]), .A2(window_bank_count[69]), 
        .A3(window_bank_count[67]), .A4(window_bank_count[64]), .Z(n522) );
  NR4D0BWP35P140 U718 ( .A1(window_bank_count[71]), .A2(window_bank_count[70]), 
        .A3(n523), .A4(n522), .ZN(n643) );
  NR2D1BWP35P140 U719 ( .A1(window_bank_count[107]), .A2(
        window_bank_count[106]), .ZN(n526) );
  NR2D1BWP35P140 U720 ( .A1(window_bank_count[111]), .A2(
        window_bank_count[110]), .ZN(n525) );
  NR4D0BWP35P140 U721 ( .A1(window_bank_count[105]), .A2(
        window_bank_count[104]), .A3(window_bank_count[109]), .A4(
        window_bank_count[108]), .ZN(n524) );
  ND3D1BWP35P140 U722 ( .A1(n526), .A2(n525), .A3(n524), .ZN(n730) );
  NR4D0BWP35P140 U724 ( .A1(window_bank_count[88]), .A2(window_bank_count[89]), 
        .A3(window_bank_count[90]), .A4(window_bank_count[94]), .ZN(n527) );
  IND3D1BWP35P140 U725 ( .A1(n529), .B1(n528), .B2(n527), .ZN(n655) );
  NR2D1BWP35P140 U726 ( .A1(window_bank_count[87]), .A2(window_bank_count[85]), 
        .ZN(n531) );
  NR4D0BWP35P140 U727 ( .A1(window_bank_count[83]), .A2(window_bank_count[86]), 
        .A3(window_bank_count[81]), .A4(window_bank_count[80]), .ZN(n530) );
  ND3D1BWP35P140 U728 ( .A1(n532), .A2(n531), .A3(n530), .ZN(n616) );
  NR2D1BWP35P140 U729 ( .A1(window_bank_count[72]), .A2(n533), .ZN(n668) );
  CKND0BWP35P140 U730 ( .I(n668), .ZN(n666) );
  NR4D0BWP35P140 U731 ( .A1(n730), .A2(n655), .A3(n616), .A4(n666), .ZN(n571)
         );
  CKND0BWP35P140 U732 ( .I(n643), .ZN(n642) );
  CKND0BWP35P140 U733 ( .I(n534), .ZN(n535) );
  INR4D0BWP35P140 U734 ( .A1(n536), .B1(n535), .B2(window_bank_count[126]), 
        .B3(window_bank_count[120]), .ZN(n625) );
  CKND0BWP35P140 U735 ( .I(n625), .ZN(n623) );
  NR4D0BWP35P140 U736 ( .A1(window_bank_count[96]), .A2(window_bank_count[103]), .A3(window_bank_count[102]), .A4(window_bank_count[97]), .ZN(n537) );
  NR2D1BWP35P140 U737 ( .A1(window_bank_count[119]), .A2(
        window_bank_count[117]), .ZN(n540) );
  NR4D0BWP35P140 U738 ( .A1(window_bank_count[112]), .A2(
        window_bank_count[113]), .A3(window_bank_count[114]), .A4(
        window_bank_count[118]), .ZN(n539) );
  NR4D0BWP35P140 U740 ( .A1(n642), .A2(n623), .A3(n734), .A4(n631), .ZN(n570)
         );
  CKND0BWP35P140 U741 ( .I(window_token_tag[21]), .ZN(n704) );
  CKND0BWP35P140 U742 ( .I(window_token_tag[23]), .ZN(n718) );
  OAI22D1BWP35P140 U743 ( .A1(n704), .A2(window_token_tag[45]), .B1(n718), 
        .B2(window_token_tag[47]), .ZN(n542) );
  AOI221D1BWP35P140 U744 ( .A1(n704), .A2(window_token_tag[45]), .B1(
        window_token_tag[47]), .B2(n718), .C(n542), .ZN(n549) );
  CKND0BWP35P140 U745 ( .I(window_token_tag[22]), .ZN(n678) );
  CKND0BWP35P140 U746 ( .I(window_token_tag[18]), .ZN(n722) );
  OAI22D1BWP35P140 U747 ( .A1(n678), .A2(window_token_tag[46]), .B1(n722), 
        .B2(window_token_tag[42]), .ZN(n543) );
  AOI221D1BWP35P140 U748 ( .A1(n678), .A2(window_token_tag[46]), .B1(
        window_token_tag[42]), .B2(n722), .C(n543), .ZN(n548) );
  CKND0BWP35P140 U749 ( .I(window_token_tag[20]), .ZN(n706) );
  CKND0BWP35P140 U750 ( .I(window_token_tag[19]), .ZN(n727) );
  OAI22D1BWP35P140 U751 ( .A1(n706), .A2(window_token_tag[44]), .B1(n727), 
        .B2(window_token_tag[43]), .ZN(n544) );
  AOI221D1BWP35P140 U752 ( .A1(n706), .A2(window_token_tag[44]), .B1(
        window_token_tag[43]), .B2(n727), .C(n544), .ZN(n547) );
  CKND0BWP35P140 U753 ( .I(window_token_tag[15]), .ZN(n716) );
  CKND0BWP35P140 U754 ( .I(window_token_tag[17]), .ZN(n708) );
  OAI22D1BWP35P140 U755 ( .A1(n716), .A2(window_token_tag[39]), .B1(n708), 
        .B2(window_token_tag[41]), .ZN(n545) );
  ND4D0BWP35P140 U757 ( .A1(n549), .A2(n548), .A3(n547), .A4(n546), .ZN(n568)
         );
  CKND0BWP35P140 U758 ( .I(window_token_tag[16]), .ZN(n714) );
  CKND0BWP35P140 U759 ( .I(window_token_tag[12]), .ZN(n712) );
  OAI22D1BWP35P140 U760 ( .A1(n714), .A2(window_token_tag[40]), .B1(n712), 
        .B2(window_token_tag[36]), .ZN(n550) );
  AOI221D1BWP35P140 U761 ( .A1(n714), .A2(window_token_tag[40]), .B1(
        window_token_tag[36]), .B2(n712), .C(n550), .ZN(n557) );
  CKND0BWP35P140 U762 ( .I(window_token_tag[14]), .ZN(n720) );
  CKND0BWP35P140 U763 ( .I(window_token_tag[13]), .ZN(n694) );
  OAI22D1BWP35P140 U764 ( .A1(n720), .A2(window_token_tag[38]), .B1(n694), 
        .B2(window_token_tag[37]), .ZN(n551) );
  AOI221D1BWP35P140 U765 ( .A1(n720), .A2(window_token_tag[38]), .B1(
        window_token_tag[37]), .B2(n694), .C(n551), .ZN(n556) );
  CKND0BWP35P140 U766 ( .I(window_token_tag[9]), .ZN(n698) );
  CKND0BWP35P140 U767 ( .I(window_token_tag[11]), .ZN(n710) );
  OAI22D1BWP35P140 U768 ( .A1(n698), .A2(window_token_tag[33]), .B1(n710), 
        .B2(window_token_tag[35]), .ZN(n552) );
  AOI221D1BWP35P140 U769 ( .A1(n698), .A2(window_token_tag[33]), .B1(
        window_token_tag[35]), .B2(n710), .C(n552), .ZN(n555) );
  CKND0BWP35P140 U770 ( .I(window_token_tag[10]), .ZN(n692) );
  CKND0BWP35P140 U771 ( .I(window_token_tag[6]), .ZN(n696) );
  OAI22D1BWP35P140 U772 ( .A1(n692), .A2(window_token_tag[34]), .B1(n696), 
        .B2(window_token_tag[30]), .ZN(n553) );
  AOI221D1BWP35P140 U773 ( .A1(n692), .A2(window_token_tag[34]), .B1(
        window_token_tag[30]), .B2(n696), .C(n553), .ZN(n554) );
  ND4D0BWP35P140 U774 ( .A1(n557), .A2(n556), .A3(n555), .A4(n554), .ZN(n567)
         );
  CKND0BWP35P140 U775 ( .I(window_token_tag[3]), .ZN(n676) );
  CKND0BWP35P140 U776 ( .I(window_token_tag[5]), .ZN(n684) );
  OAI22D1BWP35P140 U777 ( .A1(n676), .A2(window_token_tag[27]), .B1(n684), 
        .B2(window_token_tag[29]), .ZN(n558) );
  AOI221D1BWP35P140 U778 ( .A1(n676), .A2(window_token_tag[27]), .B1(
        window_token_tag[29]), .B2(n684), .C(n558), .ZN(n565) );
  CKND0BWP35P140 U779 ( .I(window_token_tag[8]), .ZN(n700) );
  CKND0BWP35P140 U780 ( .I(window_token_tag[7]), .ZN(n702) );
  OAI22D1BWP35P140 U781 ( .A1(n700), .A2(window_token_tag[32]), .B1(n702), 
        .B2(window_token_tag[31]), .ZN(n559) );
  AOI221D1BWP35P140 U782 ( .A1(n700), .A2(window_token_tag[32]), .B1(
        window_token_tag[31]), .B2(n702), .C(n559), .ZN(n564) );
  CKND0BWP35P140 U783 ( .I(window_token_tag[2]), .ZN(n686) );
  CKND0BWP35P140 U784 ( .I(window_token_tag[1]), .ZN(n688) );
  OAI22D1BWP35P140 U785 ( .A1(n686), .A2(window_token_tag[26]), .B1(n688), 
        .B2(window_token_tag[25]), .ZN(n560) );
  AOI221D1BWP35P140 U786 ( .A1(n686), .A2(window_token_tag[26]), .B1(
        window_token_tag[25]), .B2(n688), .C(n560), .ZN(n563) );
  CKND0BWP35P140 U787 ( .I(window_token_tag[4]), .ZN(n682) );
  CKND0BWP35P140 U788 ( .I(window_token_tag[0]), .ZN(n680) );
  ND4D0BWP35P140 U791 ( .A1(n565), .A2(n564), .A3(n563), .A4(n562), .ZN(n566)
         );
  OAI31D0BWP35P140 U792 ( .A1(n568), .A2(n567), .A3(n566), .B(window_valid[1]), 
        .ZN(n569) );
  CKND0BWP35P140 U793 ( .I(n730), .ZN(n731) );
  INR3D0BWP35P140 U794 ( .A1(window_head_channel[157]), .B1(
        window_head_channel[158]), .B2(window_head_channel[156]), .ZN(n575) );
  OAI31D0BWP35P140 U795 ( .A1(window_head_channel[180]), .A2(
        window_head_channel[181]), .A3(window_head_channel[182]), .B(n623), 
        .ZN(n573) );
  CKND0BWP35P140 U796 ( .I(n576), .ZN(n729) );
  INR3D0BWP35P140 U797 ( .A1(window_head_channel[61]), .B1(
        window_head_channel[62]), .B2(window_head_channel[60]), .ZN(n590) );
  IND3D1BWP35P140 U798 ( .A1(window_head_channel[25]), .B1(
        window_head_channel[26]), .B2(window_head_channel[24]), .ZN(n586) );
  CKND0BWP35P140 U799 ( .I(n577), .ZN(n641) );
  CKND0BWP35P140 U800 ( .I(n655), .ZN(n656) );
  INR3D0BWP35P140 U801 ( .A1(window_head_channel[134]), .B1(
        window_head_channel[133]), .B2(window_head_channel[132]), .ZN(n583) );
  IND3D1BWP35P140 U802 ( .A1(window_head_channel[121]), .B1(
        window_head_channel[122]), .B2(window_head_channel[120]), .ZN(n579) );
  IND3D1BWP35P140 U803 ( .A1(window_head_channel[50]), .B1(
        window_head_channel[49]), .B2(window_head_channel[48]), .ZN(n578) );
  CKND0BWP35P140 U804 ( .I(window_head_channel[168]), .ZN(n580) );
  OAI31D0BWP35P140 U805 ( .A1(window_head_channel[170]), .A2(
        window_head_channel[169]), .A3(n580), .B(n631), .ZN(n581) );
  NR4D0BWP35P140 U806 ( .A1(n594), .A2(n593), .A3(n592), .A4(n591), .ZN(n595)
         );
  INR2D1BWP35P140 U808 ( .A1(issue_valid), .B1(issue_ready), .ZN(n599) );
  NR3D0P7BWP35P140 U809 ( .A1(n600), .A2(fault_q), .A3(n599), .ZN(pair_ready)
         );
  IAO21D1BWP35P140 U810 ( .A1(n599), .A2(pair_accept), .B(rst_core), .ZN(n452)
         );
  CKND0BWP35P140 U812 ( .I(n728), .ZN(n826) );
  ND2D0BWP35P140 U813 ( .A1(n613), .A2(n614), .ZN(n603) );
  NR2D0BWP35P140 U814 ( .A1(n601), .A2(n603), .ZN(n607) );
  AOI21D0BWP35P140 U815 ( .A1(n601), .A2(n603), .B(n607), .ZN(n766) );
  CKND0BWP35P140 U816 ( .I(n766), .ZN(n763) );
  INR2D1BWP35P140 U817 ( .A1(n603), .B1(n602), .ZN(n611) );
  FA1D0BWP35P140 U818 ( .A(n813), .B(n809), .CI(n806), .CO(n770), .S(n604) );
  CKND0BWP35P140 U819 ( .I(n604), .ZN(n610) );
  FA1D0BWP35P140 U820 ( .A(n810), .B(n808), .CI(n807), .CO(n601), .S(n605) );
  CKND0BWP35P140 U821 ( .I(n605), .ZN(n609) );
  CKND0BWP35P140 U822 ( .I(n764), .ZN(n765) );
  MAOI222D0BWP35P140 U823 ( .A(n770), .B(n763), .C(n765), .ZN(n606) );
  AN2D0BWP35P140 U824 ( .A1(n828), .A2(pair_accept), .Z(n767) );
  IND3D1BWP35P140 U825 ( .A1(n770), .B1(n607), .B2(n764), .ZN(n805) );
  OAI211D0BWP35P140 U826 ( .A1(n607), .A2(n606), .B(n767), .C(n805), .ZN(n608)
         );
  IOA21D0BWP35P140 U827 ( .A1(n826), .A2(issue_source_count[2]), .B(n608), 
        .ZN(n328) );
  FA1D0BWP35P140 U828 ( .A(n611), .B(n610), .CI(n609), .CO(n764), .S(n612) );
  AO22D0BWP35P140 U829 ( .A1(n826), .A2(issue_source_count[0]), .B1(n767), 
        .B2(n612), .Z(n326) );
  CKND0BWP35P140 U830 ( .I(n728), .ZN(n811) );
  AO22D0BWP35P140 U831 ( .A1(issue_bank_valid[3]), .A2(n811), .B1(n767), .B2(
        n613), .Z(n321) );
  AO22D0BWP35P140 U832 ( .A1(issue_bank_valid[5]), .A2(n811), .B1(n767), .B2(
        n614), .Z(n323) );
  INR3D0BWP35P140 U833 ( .A1(n615), .B1(n812), .B2(n616), .ZN(n818) );
  AN2D0BWP35P140 U834 ( .A1(n616), .A2(n767), .Z(n819) );
  AOI222D0BWP35P140 U835 ( .A1(n776), .A2(issue_source_channel[28]), .B1(n818), 
        .B2(window_head_channel[28]), .C1(n819), .C2(window_head_channel[124]), 
        .ZN(n617) );
  CKND0BWP35P140 U836 ( .I(n617), .ZN(n422) );
  AOI222D0BWP35P140 U837 ( .A1(n776), .A2(issue_source_channel[30]), .B1(n818), 
        .B2(window_head_channel[30]), .C1(n819), .C2(window_head_channel[126]), 
        .ZN(n618) );
  CKND0BWP35P140 U838 ( .I(n618), .ZN(n420) );
  AOI222D0BWP35P140 U839 ( .A1(n776), .A2(issue_source_channel[32]), .B1(n818), 
        .B2(window_head_channel[32]), .C1(n819), .C2(window_head_channel[128]), 
        .ZN(n619) );
  CKND0BWP35P140 U840 ( .I(n619), .ZN(n418) );
  AOI222D0BWP35P140 U841 ( .A1(n776), .A2(issue_source_channel[34]), .B1(n818), 
        .B2(window_head_channel[34]), .C1(n819), .C2(window_head_channel[130]), 
        .ZN(n620) );
  CKND0BWP35P140 U842 ( .I(n620), .ZN(n416) );
  AOI222D0BWP35P140 U843 ( .A1(n776), .A2(issue_source_channel[35]), .B1(n818), 
        .B2(window_head_channel[35]), .C1(n819), .C2(window_head_channel[131]), 
        .ZN(n621) );
  CKND0BWP35P140 U844 ( .I(n621), .ZN(n415) );
  AOI222D0BWP35P140 U845 ( .A1(n776), .A2(issue_source_channel[31]), .B1(n818), 
        .B2(window_head_channel[31]), .C1(n819), .C2(window_head_channel[127]), 
        .ZN(n622) );
  CKND0BWP35P140 U846 ( .I(n622), .ZN(n419) );
  INR3D0BWP35P140 U847 ( .A1(n624), .B1(n812), .B2(n623), .ZN(n772) );
  NR2D0BWP35P140 U848 ( .A1(n625), .A2(n812), .ZN(n639) );
  AOI222D0BWP35P140 U849 ( .A1(n811), .A2(issue_source_channel[94]), .B1(n772), 
        .B2(window_head_channel[94]), .C1(n639), .C2(window_head_channel[190]), 
        .ZN(n626) );
  CKND0BWP35P140 U850 ( .I(n626), .ZN(n356) );
  AOI222D0BWP35P140 U851 ( .A1(n811), .A2(issue_source_channel[90]), .B1(n772), 
        .B2(window_head_channel[90]), .C1(n639), .C2(window_head_channel[186]), 
        .ZN(n627) );
  CKND0BWP35P140 U852 ( .I(n627), .ZN(n360) );
  AOI222D0BWP35P140 U853 ( .A1(n811), .A2(issue_source_channel[92]), .B1(n772), 
        .B2(window_head_channel[92]), .C1(n639), .C2(window_head_channel[188]), 
        .ZN(n628) );
  CKND0BWP35P140 U854 ( .I(n628), .ZN(n358) );
  AOI222D0BWP35P140 U855 ( .A1(n811), .A2(issue_source_channel[88]), .B1(n772), 
        .B2(window_head_channel[88]), .C1(n639), .C2(window_head_channel[184]), 
        .ZN(n629) );
  CKND0BWP35P140 U856 ( .I(n629), .ZN(n362) );
  INR3D0BWP35P140 U857 ( .A1(n630), .B1(n812), .B2(n631), .ZN(n816) );
  AN2D0BWP35P140 U858 ( .A1(n631), .A2(n767), .Z(n817) );
  AOI222D0BWP35P140 U859 ( .A1(n811), .A2(issue_source_channel[81]), .B1(n816), 
        .B2(window_head_channel[81]), .C1(n817), .C2(window_head_channel[177]), 
        .ZN(n632) );
  CKND0BWP35P140 U860 ( .I(n632), .ZN(n369) );
  AOI222D0BWP35P140 U861 ( .A1(n811), .A2(issue_source_channel[83]), .B1(n816), 
        .B2(window_head_channel[83]), .C1(n817), .C2(window_head_channel[179]), 
        .ZN(n633) );
  CKND0BWP35P140 U862 ( .I(n633), .ZN(n367) );
  CKND0BWP35P140 U863 ( .I(n728), .ZN(n723) );
  AOI222D0BWP35P140 U864 ( .A1(n723), .A2(issue_source_channel[91]), .B1(n772), 
        .B2(window_head_channel[91]), .C1(n639), .C2(window_head_channel[187]), 
        .ZN(n634) );
  CKND0BWP35P140 U865 ( .I(n634), .ZN(n359) );
  AOI222D0BWP35P140 U866 ( .A1(n723), .A2(issue_source_channel[93]), .B1(n772), 
        .B2(window_head_channel[93]), .C1(n639), .C2(window_head_channel[189]), 
        .ZN(n635) );
  CKND0BWP35P140 U867 ( .I(n635), .ZN(n357) );
  AOI222D0BWP35P140 U868 ( .A1(n723), .A2(issue_source_channel[89]), .B1(n772), 
        .B2(window_head_channel[89]), .C1(n639), .C2(window_head_channel[185]), 
        .ZN(n636) );
  CKND0BWP35P140 U869 ( .I(n636), .ZN(n361) );
  AOI222D0BWP35P140 U870 ( .A1(n723), .A2(issue_source_channel[82]), .B1(n816), 
        .B2(window_head_channel[82]), .C1(n817), .C2(window_head_channel[178]), 
        .ZN(n637) );
  CKND0BWP35P140 U871 ( .I(n637), .ZN(n368) );
  AOI222D0BWP35P140 U872 ( .A1(n723), .A2(issue_source_channel[95]), .B1(n772), 
        .B2(window_head_channel[95]), .C1(n639), .C2(window_head_channel[191]), 
        .ZN(n638) );
  CKND0BWP35P140 U873 ( .I(n638), .ZN(n355) );
  AOI222D0BWP35P140 U874 ( .A1(n723), .A2(issue_source_channel[87]), .B1(n772), 
        .B2(window_head_channel[87]), .C1(n639), .C2(window_head_channel[183]), 
        .ZN(n640) );
  CKND0BWP35P140 U875 ( .I(n640), .ZN(n363) );
  NR3D0BWP35P140 U876 ( .A1(n812), .A2(n642), .A3(n641), .ZN(n774) );
  NR2D0BWP35P140 U877 ( .A1(n643), .A2(n812), .ZN(n773) );
  AOI222D0BWP35P140 U878 ( .A1(n826), .A2(issue_source_channel[10]), .B1(n774), 
        .B2(window_head_channel[10]), .C1(n773), .C2(window_head_channel[106]), 
        .ZN(n644) );
  CKND0BWP35P140 U879 ( .I(n644), .ZN(n440) );
  AOI222D0BWP35P140 U880 ( .A1(n826), .A2(issue_source_channel[5]), .B1(n774), 
        .B2(window_head_channel[5]), .C1(n773), .C2(window_head_channel[101]), 
        .ZN(n645) );
  CKND0BWP35P140 U881 ( .I(n645), .ZN(n445) );
  AOI222D0BWP35P140 U882 ( .A1(n826), .A2(issue_source_channel[6]), .B1(n774), 
        .B2(window_head_channel[6]), .C1(n773), .C2(window_head_channel[102]), 
        .ZN(n646) );
  CKND0BWP35P140 U883 ( .I(n646), .ZN(n444) );
  AOI222D0BWP35P140 U884 ( .A1(n826), .A2(issue_source_channel[11]), .B1(n774), 
        .B2(window_head_channel[11]), .C1(n773), .C2(window_head_channel[107]), 
        .ZN(n647) );
  CKND0BWP35P140 U885 ( .I(n647), .ZN(n439) );
  AOI222D0BWP35P140 U886 ( .A1(n826), .A2(issue_source_channel[3]), .B1(n774), 
        .B2(window_head_channel[3]), .C1(n773), .C2(window_head_channel[99]), 
        .ZN(n648) );
  CKND0BWP35P140 U887 ( .I(n648), .ZN(n447) );
  AOI222D0BWP35P140 U888 ( .A1(n826), .A2(issue_source_channel[4]), .B1(n774), 
        .B2(window_head_channel[4]), .C1(n773), .C2(window_head_channel[100]), 
        .ZN(n649) );
  CKND0BWP35P140 U889 ( .I(n649), .ZN(n446) );
  AOI222D0BWP35P140 U890 ( .A1(n826), .A2(issue_source_channel[27]), .B1(n818), 
        .B2(window_head_channel[27]), .C1(n819), .C2(window_head_channel[123]), 
        .ZN(n650) );
  CKND0BWP35P140 U891 ( .I(n650), .ZN(n423) );
  AOI222D0BWP35P140 U892 ( .A1(n826), .A2(issue_source_channel[7]), .B1(n774), 
        .B2(window_head_channel[7]), .C1(n773), .C2(window_head_channel[103]), 
        .ZN(n651) );
  CKND0BWP35P140 U893 ( .I(n651), .ZN(n443) );
  AOI222D0BWP35P140 U894 ( .A1(n826), .A2(issue_source_channel[8]), .B1(n774), 
        .B2(window_head_channel[8]), .C1(n773), .C2(window_head_channel[104]), 
        .ZN(n652) );
  CKND0BWP35P140 U895 ( .I(n652), .ZN(n442) );
  AOI222D0BWP35P140 U896 ( .A1(n826), .A2(issue_source_channel[9]), .B1(n774), 
        .B2(window_head_channel[9]), .C1(n773), .C2(window_head_channel[105]), 
        .ZN(n653) );
  CKND0BWP35P140 U897 ( .I(n653), .ZN(n441) );
  NR3D0BWP35P140 U898 ( .A1(n812), .A2(n655), .A3(n654), .ZN(n820) );
  NR2D0BWP35P140 U899 ( .A1(n656), .A2(n812), .ZN(n821) );
  AOI222D0BWP35P140 U900 ( .A1(n776), .A2(issue_source_channel[39]), .B1(n820), 
        .B2(window_head_channel[39]), .C1(n821), .C2(window_head_channel[135]), 
        .ZN(n657) );
  CKND0BWP35P140 U901 ( .I(n657), .ZN(n411) );
  AOI222D0BWP35P140 U902 ( .A1(n776), .A2(issue_source_channel[46]), .B1(n820), 
        .B2(window_head_channel[46]), .C1(n821), .C2(window_head_channel[142]), 
        .ZN(n658) );
  CKND0BWP35P140 U903 ( .I(n658), .ZN(n404) );
  AOI222D0BWP35P140 U904 ( .A1(n776), .A2(issue_source_channel[45]), .B1(n820), 
        .B2(window_head_channel[45]), .C1(n821), .C2(window_head_channel[141]), 
        .ZN(n659) );
  CKND0BWP35P140 U905 ( .I(n659), .ZN(n405) );
  AOI222D0BWP35P140 U906 ( .A1(n776), .A2(issue_source_channel[44]), .B1(n820), 
        .B2(window_head_channel[44]), .C1(n821), .C2(window_head_channel[140]), 
        .ZN(n660) );
  CKND0BWP35P140 U907 ( .I(n660), .ZN(n406) );
  AOI222D0BWP35P140 U908 ( .A1(n776), .A2(issue_source_channel[43]), .B1(n820), 
        .B2(window_head_channel[43]), .C1(n821), .C2(window_head_channel[139]), 
        .ZN(n661) );
  CKND0BWP35P140 U909 ( .I(n661), .ZN(n407) );
  AOI222D0BWP35P140 U910 ( .A1(n776), .A2(issue_source_channel[42]), .B1(n820), 
        .B2(window_head_channel[42]), .C1(n821), .C2(window_head_channel[138]), 
        .ZN(n662) );
  CKND0BWP35P140 U911 ( .I(n662), .ZN(n408) );
  AOI222D0BWP35P140 U912 ( .A1(n776), .A2(issue_source_channel[47]), .B1(n820), 
        .B2(window_head_channel[47]), .C1(n821), .C2(window_head_channel[143]), 
        .ZN(n663) );
  CKND0BWP35P140 U913 ( .I(n663), .ZN(n403) );
  AOI222D0BWP35P140 U914 ( .A1(n776), .A2(issue_source_channel[40]), .B1(n820), 
        .B2(window_head_channel[40]), .C1(n821), .C2(window_head_channel[136]), 
        .ZN(n664) );
  CKND0BWP35P140 U915 ( .I(n664), .ZN(n410) );
  AOI222D0BWP35P140 U916 ( .A1(n776), .A2(issue_source_channel[41]), .B1(n820), 
        .B2(window_head_channel[41]), .C1(n821), .C2(window_head_channel[137]), 
        .ZN(n665) );
  CKND0BWP35P140 U917 ( .I(n665), .ZN(n409) );
  INR3D0BWP35P140 U918 ( .A1(n667), .B1(n812), .B2(n666), .ZN(n822) );
  NR2D0BWP35P140 U919 ( .A1(n668), .A2(n812), .ZN(n823) );
  AOI222D0BWP35P140 U920 ( .A1(n826), .A2(issue_source_channel[16]), .B1(n822), 
        .B2(window_head_channel[16]), .C1(n823), .C2(window_head_channel[112]), 
        .ZN(n669) );
  CKND0BWP35P140 U921 ( .I(n669), .ZN(n434) );
  AOI222D0BWP35P140 U922 ( .A1(n826), .A2(issue_source_channel[22]), .B1(n822), 
        .B2(window_head_channel[22]), .C1(n823), .C2(window_head_channel[118]), 
        .ZN(n670) );
  CKND0BWP35P140 U923 ( .I(n670), .ZN(n428) );
  AOI222D0BWP35P140 U924 ( .A1(n826), .A2(issue_source_channel[18]), .B1(n822), 
        .B2(window_head_channel[18]), .C1(n823), .C2(window_head_channel[114]), 
        .ZN(n671) );
  CKND0BWP35P140 U925 ( .I(n671), .ZN(n432) );
  AOI222D0BWP35P140 U926 ( .A1(n826), .A2(issue_source_channel[15]), .B1(n822), 
        .B2(window_head_channel[15]), .C1(n823), .C2(window_head_channel[111]), 
        .ZN(n672) );
  CKND0BWP35P140 U927 ( .I(n672), .ZN(n435) );
  AOI222D0BWP35P140 U928 ( .A1(n826), .A2(issue_source_channel[21]), .B1(n822), 
        .B2(window_head_channel[21]), .C1(n823), .C2(window_head_channel[117]), 
        .ZN(n673) );
  CKND0BWP35P140 U929 ( .I(n673), .ZN(n429) );
  ND2D0BWP35P140 U930 ( .A1(n674), .A2(n767), .ZN(n726) );
  NR2D0BWP35P140 U931 ( .A1(n674), .A2(n812), .ZN(n724) );
  AOI22D0BWP35P140 U932 ( .A1(window_token_tag[27]), .A2(n724), .B1(n811), 
        .B2(issue_token_tag[3]), .ZN(n675) );
  OAI21D0BWP35P140 U933 ( .A1(n676), .A2(n726), .B(n675), .ZN(n333) );
  AOI22D0BWP35P140 U934 ( .A1(window_token_tag[46]), .A2(n724), .B1(n811), 
        .B2(issue_token_tag[22]), .ZN(n677) );
  OAI21D0BWP35P140 U935 ( .A1(n678), .A2(n726), .B(n677), .ZN(n352) );
  AOI22D0BWP35P140 U936 ( .A1(window_token_tag[24]), .A2(n724), .B1(n811), 
        .B2(issue_token_tag[0]), .ZN(n679) );
  OAI21D0BWP35P140 U937 ( .A1(n680), .A2(n726), .B(n679), .ZN(n330) );
  AOI22D0BWP35P140 U938 ( .A1(window_token_tag[28]), .A2(n724), .B1(n811), 
        .B2(issue_token_tag[4]), .ZN(n681) );
  OAI21D0BWP35P140 U939 ( .A1(n682), .A2(n726), .B(n681), .ZN(n334) );
  AOI22D0BWP35P140 U940 ( .A1(window_token_tag[29]), .A2(n724), .B1(n811), 
        .B2(issue_token_tag[5]), .ZN(n683) );
  OAI21D0BWP35P140 U941 ( .A1(n684), .A2(n726), .B(n683), .ZN(n335) );
  AOI22D0BWP35P140 U942 ( .A1(window_token_tag[26]), .A2(n724), .B1(n811), 
        .B2(issue_token_tag[2]), .ZN(n685) );
  OAI21D0BWP35P140 U943 ( .A1(n686), .A2(n726), .B(n685), .ZN(n332) );
  AOI22D0BWP35P140 U944 ( .A1(window_token_tag[25]), .A2(n724), .B1(n811), 
        .B2(issue_token_tag[1]), .ZN(n687) );
  OAI21D0BWP35P140 U945 ( .A1(n688), .A2(n726), .B(n687), .ZN(n331) );
  AOI222D0BWP35P140 U946 ( .A1(n776), .A2(issue_source_channel[19]), .B1(n822), 
        .B2(window_head_channel[19]), .C1(n823), .C2(window_head_channel[115]), 
        .ZN(n689) );
  CKND0BWP35P140 U947 ( .I(n689), .ZN(n431) );
  AOI222D0BWP35P140 U948 ( .A1(n776), .A2(issue_source_channel[17]), .B1(n822), 
        .B2(window_head_channel[17]), .C1(n823), .C2(window_head_channel[113]), 
        .ZN(n690) );
  CKND0BWP35P140 U949 ( .I(n690), .ZN(n433) );
  AOI22D0BWP35P140 U950 ( .A1(window_token_tag[34]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[10]), .ZN(n691) );
  OAI21D0BWP35P140 U951 ( .A1(n692), .A2(n726), .B(n691), .ZN(n340) );
  AOI22D0BWP35P140 U952 ( .A1(window_token_tag[37]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[13]), .ZN(n693) );
  OAI21D0BWP35P140 U953 ( .A1(n694), .A2(n726), .B(n693), .ZN(n343) );
  AOI22D0BWP35P140 U954 ( .A1(window_token_tag[30]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[6]), .ZN(n695) );
  OAI21D0BWP35P140 U955 ( .A1(n696), .A2(n726), .B(n695), .ZN(n336) );
  AOI22D0BWP35P140 U956 ( .A1(window_token_tag[33]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[9]), .ZN(n697) );
  OAI21D0BWP35P140 U957 ( .A1(n698), .A2(n726), .B(n697), .ZN(n339) );
  AOI22D0BWP35P140 U958 ( .A1(window_token_tag[32]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[8]), .ZN(n699) );
  OAI21D0BWP35P140 U959 ( .A1(n700), .A2(n726), .B(n699), .ZN(n338) );
  AOI22D0BWP35P140 U960 ( .A1(window_token_tag[31]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[7]), .ZN(n701) );
  OAI21D0BWP35P140 U961 ( .A1(n702), .A2(n726), .B(n701), .ZN(n337) );
  AOI22D0BWP35P140 U962 ( .A1(window_token_tag[45]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[21]), .ZN(n703) );
  OAI21D0BWP35P140 U963 ( .A1(n704), .A2(n726), .B(n703), .ZN(n351) );
  AOI22D0BWP35P140 U964 ( .A1(window_token_tag[44]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[20]), .ZN(n705) );
  OAI21D0BWP35P140 U965 ( .A1(n706), .A2(n726), .B(n705), .ZN(n350) );
  AOI22D0BWP35P140 U966 ( .A1(window_token_tag[41]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[17]), .ZN(n707) );
  OAI21D0BWP35P140 U967 ( .A1(n708), .A2(n726), .B(n707), .ZN(n347) );
  AOI22D0BWP35P140 U968 ( .A1(window_token_tag[35]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[11]), .ZN(n709) );
  OAI21D0BWP35P140 U969 ( .A1(n710), .A2(n726), .B(n709), .ZN(n341) );
  AOI22D0BWP35P140 U970 ( .A1(window_token_tag[36]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[12]), .ZN(n711) );
  OAI21D0BWP35P140 U971 ( .A1(n712), .A2(n726), .B(n711), .ZN(n342) );
  AOI22D0BWP35P140 U972 ( .A1(window_token_tag[40]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[16]), .ZN(n713) );
  OAI21D0BWP35P140 U973 ( .A1(n714), .A2(n726), .B(n713), .ZN(n346) );
  AOI22D0BWP35P140 U974 ( .A1(window_token_tag[39]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[15]), .ZN(n715) );
  OAI21D0BWP35P140 U975 ( .A1(n716), .A2(n726), .B(n715), .ZN(n345) );
  AOI22D0BWP35P140 U976 ( .A1(window_token_tag[47]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[23]), .ZN(n717) );
  OAI21D0BWP35P140 U977 ( .A1(n718), .A2(n726), .B(n717), .ZN(n353) );
  AOI22D0BWP35P140 U978 ( .A1(window_token_tag[38]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[14]), .ZN(n719) );
  OAI21D0BWP35P140 U979 ( .A1(n720), .A2(n726), .B(n719), .ZN(n344) );
  AOI22D0BWP35P140 U980 ( .A1(window_token_tag[42]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[18]), .ZN(n721) );
  OAI21D0BWP35P140 U981 ( .A1(n722), .A2(n726), .B(n721), .ZN(n348) );
  AOI22D0BWP35P140 U982 ( .A1(window_token_tag[43]), .A2(n724), .B1(n723), 
        .B2(issue_token_tag[19]), .ZN(n725) );
  OAI21D0BWP35P140 U983 ( .A1(n727), .A2(n726), .B(n725), .ZN(n349) );
  CKND0BWP35P140 U984 ( .I(n776), .ZN(n728) );
  CKND0BWP35P140 U985 ( .I(n728), .ZN(n761) );
  NR3D0BWP35P140 U986 ( .A1(n812), .A2(n730), .A3(n729), .ZN(n824) );
  NR2D0BWP35P140 U987 ( .A1(n731), .A2(n812), .ZN(n825) );
  AOI222D0BWP35P140 U988 ( .A1(n761), .A2(issue_source_channel[70]), .B1(n824), 
        .B2(window_head_channel[70]), .C1(n825), .C2(window_head_channel[166]), 
        .ZN(n732) );
  CKND0BWP35P140 U989 ( .I(n732), .ZN(n380) );
  INR3D0BWP35P140 U990 ( .A1(n733), .B1(n812), .B2(n734), .ZN(n814) );
  AN2D0BWP35P140 U991 ( .A1(n734), .A2(n767), .Z(n815) );
  AOI222D0BWP35P140 U992 ( .A1(n761), .A2(issue_source_channel[56]), .B1(n814), 
        .B2(window_head_channel[56]), .C1(n815), .C2(window_head_channel[152]), 
        .ZN(n735) );
  CKND0BWP35P140 U993 ( .I(n735), .ZN(n394) );
  AOI222D0BWP35P140 U994 ( .A1(n761), .A2(issue_source_channel[79]), .B1(n816), 
        .B2(window_head_channel[79]), .C1(n817), .C2(window_head_channel[175]), 
        .ZN(n736) );
  CKND0BWP35P140 U995 ( .I(n736), .ZN(n371) );
  AOI222D0BWP35P140 U996 ( .A1(n761), .A2(issue_source_channel[80]), .B1(n816), 
        .B2(window_head_channel[80]), .C1(n817), .C2(window_head_channel[176]), 
        .ZN(n737) );
  CKND0BWP35P140 U997 ( .I(n737), .ZN(n370) );
  AOI222D0BWP35P140 U998 ( .A1(n761), .A2(issue_source_channel[75]), .B1(n816), 
        .B2(window_head_channel[75]), .C1(n817), .C2(window_head_channel[171]), 
        .ZN(n738) );
  CKND0BWP35P140 U999 ( .I(n738), .ZN(n375) );
  AOI222D0BWP35P140 U1000 ( .A1(n761), .A2(issue_source_channel[76]), .B1(n816), .B2(window_head_channel[76]), .C1(n817), .C2(window_head_channel[172]), .ZN(
        n739) );
  CKND0BWP35P140 U1001 ( .I(n739), .ZN(n374) );
  AOI222D0BWP35P140 U1002 ( .A1(n761), .A2(issue_source_channel[77]), .B1(n816), .B2(window_head_channel[77]), .C1(n817), .C2(window_head_channel[173]), .ZN(
        n740) );
  CKND0BWP35P140 U1003 ( .I(n740), .ZN(n373) );
  AOI222D0BWP35P140 U1004 ( .A1(n761), .A2(issue_source_channel[78]), .B1(n816), .B2(window_head_channel[78]), .C1(n817), .C2(window_head_channel[174]), .ZN(
        n741) );
  CKND0BWP35P140 U1005 ( .I(n741), .ZN(n372) );
  AOI222D0BWP35P140 U1006 ( .A1(n761), .A2(issue_source_channel[59]), .B1(n814), .B2(window_head_channel[59]), .C1(n815), .C2(window_head_channel[155]), .ZN(
        n742) );
  CKND0BWP35P140 U1007 ( .I(n742), .ZN(n391) );
  AOI222D0BWP35P140 U1008 ( .A1(n761), .A2(issue_source_channel[33]), .B1(n818), .B2(window_head_channel[33]), .C1(n819), .C2(window_head_channel[129]), .ZN(
        n743) );
  CKND0BWP35P140 U1009 ( .I(n743), .ZN(n417) );
  AOI222D0BWP35P140 U1010 ( .A1(n761), .A2(issue_source_channel[63]), .B1(n824), .B2(window_head_channel[63]), .C1(n825), .C2(window_head_channel[159]), .ZN(
        n744) );
  CKND0BWP35P140 U1011 ( .I(n744), .ZN(n387) );
  AOI222D0BWP35P140 U1012 ( .A1(n761), .A2(issue_source_channel[64]), .B1(n824), .B2(window_head_channel[64]), .C1(n825), .C2(window_head_channel[160]), .ZN(
        n745) );
  CKND0BWP35P140 U1013 ( .I(n745), .ZN(n386) );
  AOI222D0BWP35P140 U1014 ( .A1(n761), .A2(issue_source_channel[65]), .B1(n824), .B2(window_head_channel[65]), .C1(n825), .C2(window_head_channel[161]), .ZN(
        n746) );
  CKND0BWP35P140 U1015 ( .I(n746), .ZN(n385) );
  AOI222D0BWP35P140 U1016 ( .A1(n761), .A2(issue_source_channel[58]), .B1(n814), .B2(window_head_channel[58]), .C1(n815), .C2(window_head_channel[154]), .ZN(
        n747) );
  CKND0BWP35P140 U1017 ( .I(n747), .ZN(n392) );
  AOI222D0BWP35P140 U1018 ( .A1(n761), .A2(issue_source_channel[66]), .B1(n824), .B2(window_head_channel[66]), .C1(n825), .C2(window_head_channel[162]), .ZN(
        n748) );
  CKND0BWP35P140 U1019 ( .I(n748), .ZN(n384) );
  AOI222D0BWP35P140 U1020 ( .A1(n761), .A2(issue_source_channel[67]), .B1(n824), .B2(window_head_channel[67]), .C1(n825), .C2(window_head_channel[163]), .ZN(
        n749) );
  CKND0BWP35P140 U1021 ( .I(n749), .ZN(n383) );
  AOI222D0BWP35P140 U1022 ( .A1(n761), .A2(issue_source_channel[68]), .B1(n824), .B2(window_head_channel[68]), .C1(n825), .C2(window_head_channel[164]), .ZN(
        n750) );
  CKND0BWP35P140 U1023 ( .I(n750), .ZN(n382) );
  AOI222D0BWP35P140 U1024 ( .A1(n761), .A2(issue_source_channel[69]), .B1(n824), .B2(window_head_channel[69]), .C1(n825), .C2(window_head_channel[165]), .ZN(
        n751) );
  CKND0BWP35P140 U1025 ( .I(n751), .ZN(n381) );
  AOI222D0BWP35P140 U1026 ( .A1(n761), .A2(issue_source_channel[23]), .B1(n822), .B2(window_head_channel[23]), .C1(n823), .C2(window_head_channel[119]), .ZN(
        n752) );
  CKND0BWP35P140 U1027 ( .I(n752), .ZN(n427) );
  AOI222D0BWP35P140 U1028 ( .A1(n761), .A2(issue_source_channel[29]), .B1(n818), .B2(window_head_channel[29]), .C1(n819), .C2(window_head_channel[125]), .ZN(
        n753) );
  CKND0BWP35P140 U1029 ( .I(n753), .ZN(n421) );
  AOI222D0BWP35P140 U1030 ( .A1(n761), .A2(issue_source_channel[57]), .B1(n814), .B2(window_head_channel[57]), .C1(n815), .C2(window_head_channel[153]), .ZN(
        n754) );
  CKND0BWP35P140 U1031 ( .I(n754), .ZN(n393) );
  AOI222D0BWP35P140 U1032 ( .A1(n761), .A2(issue_source_channel[20]), .B1(n822), .B2(window_head_channel[20]), .C1(n823), .C2(window_head_channel[116]), .ZN(
        n755) );
  CKND0BWP35P140 U1033 ( .I(n755), .ZN(n430) );
  AOI222D0BWP35P140 U1034 ( .A1(n761), .A2(issue_source_channel[54]), .B1(n814), .B2(window_head_channel[54]), .C1(n815), .C2(window_head_channel[150]), .ZN(
        n756) );
  CKND0BWP35P140 U1035 ( .I(n756), .ZN(n396) );
  AOI222D0BWP35P140 U1036 ( .A1(n761), .A2(issue_source_channel[53]), .B1(n814), .B2(window_head_channel[53]), .C1(n815), .C2(window_head_channel[149]), .ZN(
        n757) );
  CKND0BWP35P140 U1037 ( .I(n757), .ZN(n397) );
  AOI222D0BWP35P140 U1038 ( .A1(n761), .A2(issue_source_channel[52]), .B1(n814), .B2(window_head_channel[52]), .C1(n815), .C2(window_head_channel[148]), .ZN(
        n758) );
  CKND0BWP35P140 U1039 ( .I(n758), .ZN(n398) );
  AOI222D0BWP35P140 U1040 ( .A1(n761), .A2(issue_source_channel[51]), .B1(n814), .B2(window_head_channel[51]), .C1(n815), .C2(window_head_channel[147]), .ZN(
        n759) );
  CKND0BWP35P140 U1041 ( .I(n759), .ZN(n399) );
  AOI222D0BWP35P140 U1042 ( .A1(n761), .A2(issue_source_channel[55]), .B1(n814), .B2(window_head_channel[55]), .C1(n815), .C2(window_head_channel[151]), .ZN(
        n760) );
  CKND0BWP35P140 U1043 ( .I(n760), .ZN(n395) );
  AOI222D0BWP35P140 U1044 ( .A1(n761), .A2(issue_source_channel[71]), .B1(n824), .B2(window_head_channel[71]), .C1(n825), .C2(window_head_channel[167]), .ZN(
        n762) );
  CKND0BWP35P140 U1045 ( .I(n762), .ZN(n379) );
  AOI22D0BWP35P140 U1046 ( .A1(n766), .A2(n765), .B1(n764), .B2(n763), .ZN(
        n769) );
  OAI21D0BWP35P140 U1047 ( .A1(n770), .A2(n769), .B(n767), .ZN(n768) );
  AOI21D0BWP35P140 U1048 ( .A1(n770), .A2(n769), .B(n768), .ZN(n771) );
  AO21D0BWP35P140 U1049 ( .A1(n776), .A2(issue_source_count[1]), .B(n771), .Z(
        n327) );
  AO21D0BWP35P140 U1050 ( .A1(n776), .A2(issue_selected_window[4]), .B(n820), 
        .Z(n314) );
  AO21D0BWP35P140 U1051 ( .A1(n776), .A2(issue_selected_window[2]), .B(n824), 
        .Z(n312) );
  AO21D0BWP35P140 U1052 ( .A1(n776), .A2(issue_selected_window[7]), .B(n774), 
        .Z(n317) );
  AO21D0BWP35P140 U1053 ( .A1(n776), .A2(issue_selected_window[0]), .B(n772), 
        .Z(n310) );
  AO21D0BWP35P140 U1054 ( .A1(n776), .A2(issue_selected_window[1]), .B(n816), 
        .Z(n311) );
  AO21D0BWP35P140 U1055 ( .A1(n776), .A2(issue_selected_window[3]), .B(n814), 
        .Z(n313) );
  AO21D0BWP35P140 U1056 ( .A1(n776), .A2(issue_selected_window[6]), .B(n822), 
        .Z(n316) );
  AO21D0BWP35P140 U1057 ( .A1(n776), .A2(issue_selected_window[5]), .B(n818), 
        .Z(n315) );
  NR2D0BWP35P140 U1058 ( .A1(n774), .A2(n773), .ZN(n775) );
  IOA21D0BWP35P140 U1059 ( .A1(n776), .A2(issue_source_channel[1]), .B(n775), 
        .ZN(n449) );
  IOA21D0BWP35P140 U1060 ( .A1(n776), .A2(issue_source_channel[0]), .B(n775), 
        .ZN(n450) );
  IOA21D0BWP35P140 U1061 ( .A1(n776), .A2(issue_source_channel[2]), .B(n775), 
        .ZN(n448) );
  ND3D0BWP35P140 U1062 ( .A1(n779), .A2(n778), .A3(n777), .ZN(n785) );
  NR4D0BWP35P140 U1063 ( .A1(window_bank_count[2]), .A2(window_bank_count[5]), 
        .A3(window_bank_count[109]), .A4(window_bank_count[51]), .ZN(n781) );
  NR4D0BWP35P140 U1064 ( .A1(window_bank_count[21]), .A2(window_bank_count[18]), .A3(window_bank_count[97]), .A4(window_bank_count[33]), .ZN(n780) );
  ND4D0BWP35P140 U1065 ( .A1(n783), .A2(n782), .A3(n781), .A4(n780), .ZN(n784)
         );
  NR4D0BWP35P140 U1066 ( .A1(window_bank_count[29]), .A2(n786), .A3(n785), 
        .A4(n784), .ZN(n802) );
  NR4D0BWP35P140 U1067 ( .A1(window_bank_count[45]), .A2(window_bank_count[42]), .A3(window_bank_count[10]), .A4(window_bank_count[59]), .ZN(n787) );
  AN4D0BWP35P140 U1068 ( .A1(n790), .A2(n789), .A3(n788), .A4(n787), .Z(n801)
         );
  IND3D1BWP35P140 U1069 ( .A1(window_bank_count[61]), .B1(n792), .B2(n791), 
        .ZN(n798) );
  IND4D1BWP35P140 U1070 ( .A1(n796), .B1(n795), .B2(n794), .B3(n793), .ZN(n797) );
  NR4D0BWP35P140 U1071 ( .A1(window_bank_count[58]), .A2(n799), .A3(n798), 
        .A4(n797), .ZN(n800) );
  ND4D0BWP35P140 U1072 ( .A1(n803), .A2(n802), .A3(n801), .A4(n800), .ZN(n804)
         );
  MOAI22D0BWP35P140 U1073 ( .A1(n812), .A2(n804), .B1(n826), .B2(
        issue_pair_last), .ZN(n451) );
  MOAI22D0BWP35P140 U1074 ( .A1(n812), .A2(n805), .B1(n826), .B2(
        issue_source_count[3]), .ZN(n329) );
  MOAI22D0BWP35P140 U1075 ( .A1(n806), .A2(n812), .B1(issue_bank_valid[6]), 
        .B2(n811), .ZN(n324) );
  MOAI22D0BWP35P140 U1076 ( .A1(n807), .A2(n812), .B1(issue_bank_valid[7]), 
        .B2(n811), .ZN(n325) );
  MOAI22D0BWP35P140 U1077 ( .A1(n808), .A2(n812), .B1(issue_bank_valid[0]), 
        .B2(n811), .ZN(n318) );
  MOAI22D0BWP35P140 U1078 ( .A1(n809), .A2(n812), .B1(issue_bank_valid[2]), 
        .B2(n811), .ZN(n320) );
  MOAI22D0BWP35P140 U1079 ( .A1(n810), .A2(n812), .B1(issue_bank_valid[1]), 
        .B2(n811), .ZN(n319) );
  MOAI22D0BWP35P140 U1080 ( .A1(n813), .A2(n812), .B1(issue_bank_valid[4]), 
        .B2(n811), .ZN(n322) );
  AO211D0BWP35P140 U1081 ( .A1(n826), .A2(issue_source_channel[48]), .B(n815), 
        .C(n814), .Z(n402) );
  AO211D0BWP35P140 U1082 ( .A1(n826), .A2(issue_source_channel[49]), .B(n815), 
        .C(n814), .Z(n401) );
  AO211D0BWP35P140 U1083 ( .A1(n826), .A2(issue_source_channel[72]), .B(n817), 
        .C(n816), .Z(n378) );
  AO211D0BWP35P140 U1084 ( .A1(n826), .A2(issue_source_channel[14]), .B(n823), 
        .C(n822), .Z(n436) );
  AO211D0BWP35P140 U1085 ( .A1(n826), .A2(issue_source_channel[24]), .B(n819), 
        .C(n818), .Z(n426) );
  AO211D0BWP35P140 U1086 ( .A1(n826), .A2(issue_source_channel[26]), .B(n819), 
        .C(n818), .Z(n424) );
  AO211D0BWP35P140 U1087 ( .A1(n826), .A2(issue_source_channel[38]), .B(n821), 
        .C(n820), .Z(n412) );
  AO211D0BWP35P140 U1088 ( .A1(n826), .A2(issue_source_channel[13]), .B(n823), 
        .C(n822), .Z(n437) );
  AO211D0BWP35P140 U1089 ( .A1(n826), .A2(issue_source_channel[61]), .B(n825), 
        .C(n824), .Z(n389) );
  DFKCNQD1BWP35P140 fault_q_reg ( .CN(n828), .D(protocol_error), .CP(clk_core), 
        .Q(fault_q) );
  DFKCNQD1BWP35P140 issue_bank_valid_q_reg_2_ ( .CN(n320), .D(n841), .CP(
        clk_core), .Q(issue_bank_valid[2]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_4__10_ ( .CN(n841), .D(n404), 
        .CP(clk_core), .Q(issue_source_channel[46]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_2__4_ ( .CN(n841), .D(n386), 
        .CP(clk_core), .Q(issue_source_channel[64]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__1_ ( .CN(n841), .D(n437), 
        .CP(clk_core), .Q(issue_source_channel[13]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_1__9_ ( .CN(n841), .D(n369), 
        .CP(clk_core), .Q(issue_source_channel[81]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__5_ ( .CN(n841), .D(n421), 
        .CP(clk_core), .Q(issue_source_channel[29]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_20_ ( .CN(n841), .D(n350), .CP(
        clk_core), .Q(issue_token_tag[20]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_5_ ( .CN(n841), .D(n335), .CP(
        clk_core), .Q(issue_token_tag[5]) );
  DFKCNQD1BWP35P140 issue_valid_q_reg ( .CN(n452), .D(n841), .CP(clk_core), 
        .Q(issue_valid) );
  DFKCNQD1BWP35P140 issue_source_count_q_reg_1_ ( .CN(n841), .D(n327), .CP(
        clk_core), .Q(issue_source_count[1]) );
  DFKCNQD1BWP35P140 issue_selected_window_q_reg_7_ ( .CN(n841), .D(n317), .CP(
        clk_core), .Q(issue_selected_window[7]) );
  DFKCNQD1BWP35P140 issue_selected_window_q_reg_4_ ( .CN(n841), .D(n314), .CP(
        clk_core), .Q(issue_selected_window[4]) );
  DFKCNQD1BWP35P140 issue_selected_window_q_reg_2_ ( .CN(n841), .D(n312), .CP(
        clk_core), .Q(issue_selected_window[2]) );
  DFKCNQD1BWP35P140 issue_bank_valid_q_reg_5_ ( .CN(n841), .D(n323), .CP(
        clk_core), .Q(issue_bank_valid[5]) );
  DFKCNQD1BWP35P140 issue_bank_valid_q_reg_3_ ( .CN(n841), .D(n321), .CP(
        clk_core), .Q(issue_bank_valid[3]) );
  DFKCNQD1BWP35P140 issue_source_count_q_reg_2_ ( .CN(n841), .D(n328), .CP(
        clk_core), .Q(issue_source_count[2]) );
  DFKCNQD1BWP35P140 issue_bank_valid_q_reg_7_ ( .CN(n841), .D(n325), .CP(
        clk_core), .Q(issue_bank_valid[7]) );
  DFKCNQD1BWP35P140 issue_bank_valid_q_reg_6_ ( .CN(n841), .D(n324), .CP(
        clk_core), .Q(issue_bank_valid[6]) );
  DFKCNQD1BWP35P140 issue_bank_valid_q_reg_4_ ( .CN(n841), .D(n322), .CP(
        clk_core), .Q(issue_bank_valid[4]) );
  DFKCNQD1BWP35P140 issue_bank_valid_q_reg_1_ ( .CN(n841), .D(n319), .CP(
        clk_core), .Q(issue_bank_valid[1]) );
  DFKCNQD1BWP35P140 issue_bank_valid_q_reg_0_ ( .CN(n841), .D(n318), .CP(
        clk_core), .Q(issue_bank_valid[0]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__2_ ( .CN(n841), .D(n448), 
        .CP(clk_core), .Q(issue_source_channel[2]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__1_ ( .CN(n841), .D(n449), 
        .CP(clk_core), .Q(issue_source_channel[1]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__0_ ( .CN(n841), .D(n450), 
        .CP(clk_core), .Q(issue_source_channel[0]) );
  DFKCNQD1BWP35P140 issue_source_count_q_reg_0_ ( .CN(n841), .D(n326), .CP(
        clk_core), .Q(issue_source_count[0]) );
  DFKCNQD1BWP35P140 issue_source_count_q_reg_3_ ( .CN(n841), .D(n329), .CP(
        clk_core), .Q(issue_source_count[3]) );
  DFKCNQD1BWP35P140 issue_pair_last_q_reg ( .CN(n841), .D(n451), .CP(clk_core), 
        .Q(issue_pair_last) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_4__2_ ( .CN(n841), .D(n412), 
        .CP(clk_core), .Q(issue_source_channel[38]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_2__1_ ( .CN(n841), .D(n389), 
        .CP(clk_core), .Q(issue_source_channel[61]) );
  DFKCNQD1BWP35P140 issue_selected_window_q_reg_0_ ( .CN(n841), .D(n310), .CP(
        clk_core), .Q(issue_selected_window[0]) );
  DFKCNQD1BWP35P140 issue_selected_window_q_reg_1_ ( .CN(n841), .D(n311), .CP(
        clk_core), .Q(issue_selected_window[1]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_4__11_ ( .CN(n841), .D(n403), 
        .CP(clk_core), .Q(issue_source_channel[47]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_4__9_ ( .CN(n841), .D(n405), 
        .CP(clk_core), .Q(issue_source_channel[45]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_4__8_ ( .CN(n841), .D(n406), 
        .CP(clk_core), .Q(issue_source_channel[44]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_4__7_ ( .CN(n841), .D(n407), 
        .CP(clk_core), .Q(issue_source_channel[43]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_4__6_ ( .CN(n841), .D(n408), 
        .CP(clk_core), .Q(issue_source_channel[42]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_4__5_ ( .CN(n841), .D(n409), 
        .CP(clk_core), .Q(issue_source_channel[41]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_4__4_ ( .CN(n841), .D(n410), 
        .CP(clk_core), .Q(issue_source_channel[40]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_4__3_ ( .CN(n841), .D(n411), 
        .CP(clk_core), .Q(issue_source_channel[39]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__11_ ( .CN(n841), .D(n439), 
        .CP(clk_core), .Q(issue_source_channel[11]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__10_ ( .CN(n841), .D(n440), 
        .CP(clk_core), .Q(issue_source_channel[10]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__9_ ( .CN(n841), .D(n441), 
        .CP(clk_core), .Q(issue_source_channel[9]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__8_ ( .CN(n841), .D(n442), 
        .CP(clk_core), .Q(issue_source_channel[8]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__7_ ( .CN(n841), .D(n443), 
        .CP(clk_core), .Q(issue_source_channel[7]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__6_ ( .CN(n841), .D(n444), 
        .CP(clk_core), .Q(issue_source_channel[6]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__5_ ( .CN(n841), .D(n445), 
        .CP(clk_core), .Q(issue_source_channel[5]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__4_ ( .CN(n841), .D(n446), 
        .CP(clk_core), .Q(issue_source_channel[4]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_7__3_ ( .CN(n841), .D(n447), 
        .CP(clk_core), .Q(issue_source_channel[3]) );
  DFKCNQD1BWP35P140 issue_selected_window_q_reg_6_ ( .CN(n841), .D(n316), .CP(
        clk_core), .Q(issue_selected_window[6]) );
  DFKCNQD1BWP35P140 issue_selected_window_q_reg_5_ ( .CN(n841), .D(n315), .CP(
        clk_core), .Q(issue_selected_window[5]) );
  DFKCNQD1BWP35P140 issue_selected_window_q_reg_3_ ( .CN(n841), .D(n313), .CP(
        clk_core), .Q(issue_selected_window[3]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_1__0_ ( .CN(n841), .D(n378), 
        .CP(clk_core), .Q(issue_source_channel[72]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_2__11_ ( .CN(n841), .D(n379), 
        .CP(clk_core), .Q(issue_source_channel[71]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_2__10_ ( .CN(n841), .D(n380), 
        .CP(clk_core), .Q(issue_source_channel[70]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_2__9_ ( .CN(n841), .D(n381), 
        .CP(clk_core), .Q(issue_source_channel[69]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_2__8_ ( .CN(n841), .D(n382), 
        .CP(clk_core), .Q(issue_source_channel[68]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_2__7_ ( .CN(n841), .D(n383), 
        .CP(clk_core), .Q(issue_source_channel[67]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_2__6_ ( .CN(n841), .D(n384), 
        .CP(clk_core), .Q(issue_source_channel[66]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_2__5_ ( .CN(n841), .D(n385), 
        .CP(clk_core), .Q(issue_source_channel[65]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_2__3_ ( .CN(n841), .D(n387), 
        .CP(clk_core), .Q(issue_source_channel[63]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__2_ ( .CN(n841), .D(n436), 
        .CP(clk_core), .Q(issue_source_channel[14]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__2_ ( .CN(n841), .D(n424), 
        .CP(clk_core), .Q(issue_source_channel[26]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__0_ ( .CN(n841), .D(n426), 
        .CP(clk_core), .Q(issue_source_channel[24]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__1_ ( .CN(n841), .D(n401), 
        .CP(clk_core), .Q(issue_source_channel[49]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__0_ ( .CN(n841), .D(n402), 
        .CP(clk_core), .Q(issue_source_channel[48]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_0__11_ ( .CN(n841), .D(n355), 
        .CP(clk_core), .Q(issue_source_channel[95]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_0__10_ ( .CN(n841), .D(n356), 
        .CP(clk_core), .Q(issue_source_channel[94]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_0__9_ ( .CN(n841), .D(n357), 
        .CP(clk_core), .Q(issue_source_channel[93]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_0__8_ ( .CN(n841), .D(n358), 
        .CP(clk_core), .Q(issue_source_channel[92]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_0__7_ ( .CN(n841), .D(n359), 
        .CP(clk_core), .Q(issue_source_channel[91]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_0__6_ ( .CN(n841), .D(n360), 
        .CP(clk_core), .Q(issue_source_channel[90]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_0__5_ ( .CN(n841), .D(n361), 
        .CP(clk_core), .Q(issue_source_channel[89]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_0__4_ ( .CN(n841), .D(n362), 
        .CP(clk_core), .Q(issue_source_channel[88]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_0__3_ ( .CN(n841), .D(n363), 
        .CP(clk_core), .Q(issue_source_channel[87]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_1__11_ ( .CN(n841), .D(n367), 
        .CP(clk_core), .Q(issue_source_channel[83]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_1__10_ ( .CN(n841), .D(n368), 
        .CP(clk_core), .Q(issue_source_channel[82]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_1__8_ ( .CN(n841), .D(n370), 
        .CP(clk_core), .Q(issue_source_channel[80]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_1__7_ ( .CN(n841), .D(n371), 
        .CP(clk_core), .Q(issue_source_channel[79]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_1__6_ ( .CN(n841), .D(n372), 
        .CP(clk_core), .Q(issue_source_channel[78]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_1__5_ ( .CN(n841), .D(n373), 
        .CP(clk_core), .Q(issue_source_channel[77]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_1__4_ ( .CN(n841), .D(n374), 
        .CP(clk_core), .Q(issue_source_channel[76]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_1__3_ ( .CN(n841), .D(n375), 
        .CP(clk_core), .Q(issue_source_channel[75]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__11_ ( .CN(n841), .D(n427), 
        .CP(clk_core), .Q(issue_source_channel[23]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__10_ ( .CN(n841), .D(n428), 
        .CP(clk_core), .Q(issue_source_channel[22]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__9_ ( .CN(n841), .D(n429), 
        .CP(clk_core), .Q(issue_source_channel[21]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__8_ ( .CN(n841), .D(n430), 
        .CP(clk_core), .Q(issue_source_channel[20]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__6_ ( .CN(n841), .D(n432), 
        .CP(clk_core), .Q(issue_source_channel[18]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__4_ ( .CN(n841), .D(n434), 
        .CP(clk_core), .Q(issue_source_channel[16]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__3_ ( .CN(n841), .D(n435), 
        .CP(clk_core), .Q(issue_source_channel[15]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__9_ ( .CN(n841), .D(n417), 
        .CP(clk_core), .Q(issue_source_channel[33]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__3_ ( .CN(n841), .D(n423), 
        .CP(clk_core), .Q(issue_source_channel[27]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__11_ ( .CN(n841), .D(n391), 
        .CP(clk_core), .Q(issue_source_channel[59]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__10_ ( .CN(n841), .D(n392), 
        .CP(clk_core), .Q(issue_source_channel[58]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__9_ ( .CN(n841), .D(n393), 
        .CP(clk_core), .Q(issue_source_channel[57]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__8_ ( .CN(n841), .D(n394), 
        .CP(clk_core), .Q(issue_source_channel[56]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__7_ ( .CN(n841), .D(n395), 
        .CP(clk_core), .Q(issue_source_channel[55]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__6_ ( .CN(n841), .D(n396), 
        .CP(clk_core), .Q(issue_source_channel[54]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__5_ ( .CN(n841), .D(n397), 
        .CP(clk_core), .Q(issue_source_channel[53]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__4_ ( .CN(n841), .D(n398), 
        .CP(clk_core), .Q(issue_source_channel[52]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_3__3_ ( .CN(n841), .D(n399), 
        .CP(clk_core), .Q(issue_source_channel[51]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__7_ ( .CN(n841), .D(n431), 
        .CP(clk_core), .Q(issue_source_channel[19]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_6__5_ ( .CN(n841), .D(n433), 
        .CP(clk_core), .Q(issue_source_channel[17]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__11_ ( .CN(n841), .D(n415), 
        .CP(clk_core), .Q(issue_source_channel[35]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__10_ ( .CN(n841), .D(n416), 
        .CP(clk_core), .Q(issue_source_channel[34]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__8_ ( .CN(n841), .D(n418), 
        .CP(clk_core), .Q(issue_source_channel[32]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__7_ ( .CN(n841), .D(n419), 
        .CP(clk_core), .Q(issue_source_channel[31]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__6_ ( .CN(n841), .D(n420), 
        .CP(clk_core), .Q(issue_source_channel[30]) );
  DFKCNQD1BWP35P140 issue_source_channel_q_reg_5__4_ ( .CN(n841), .D(n422), 
        .CP(clk_core), .Q(issue_source_channel[28]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_23_ ( .CN(n841), .D(n353), .CP(
        clk_core), .Q(issue_token_tag[23]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_22_ ( .CN(n841), .D(n352), .CP(
        clk_core), .Q(issue_token_tag[22]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_21_ ( .CN(n841), .D(n351), .CP(
        clk_core), .Q(issue_token_tag[21]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_19_ ( .CN(n841), .D(n349), .CP(
        clk_core), .Q(issue_token_tag[19]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_18_ ( .CN(n841), .D(n348), .CP(
        clk_core), .Q(issue_token_tag[18]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_17_ ( .CN(n841), .D(n347), .CP(
        clk_core), .Q(issue_token_tag[17]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_16_ ( .CN(n841), .D(n346), .CP(
        clk_core), .Q(issue_token_tag[16]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_15_ ( .CN(n841), .D(n345), .CP(
        clk_core), .Q(issue_token_tag[15]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_14_ ( .CN(n841), .D(n344), .CP(
        clk_core), .Q(issue_token_tag[14]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_13_ ( .CN(n841), .D(n343), .CP(
        clk_core), .Q(issue_token_tag[13]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_12_ ( .CN(n841), .D(n342), .CP(
        clk_core), .Q(issue_token_tag[12]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_11_ ( .CN(n841), .D(n341), .CP(
        clk_core), .Q(issue_token_tag[11]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_10_ ( .CN(n841), .D(n340), .CP(
        clk_core), .Q(issue_token_tag[10]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_9_ ( .CN(n841), .D(n339), .CP(
        clk_core), .Q(issue_token_tag[9]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_8_ ( .CN(n841), .D(n338), .CP(
        clk_core), .Q(issue_token_tag[8]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_7_ ( .CN(n841), .D(n337), .CP(
        clk_core), .Q(issue_token_tag[7]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_6_ ( .CN(n841), .D(n336), .CP(
        clk_core), .Q(issue_token_tag[6]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_4_ ( .CN(n841), .D(n334), .CP(
        clk_core), .Q(issue_token_tag[4]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_3_ ( .CN(n841), .D(n333), .CP(
        clk_core), .Q(issue_token_tag[3]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_2_ ( .CN(n841), .D(n332), .CP(
        clk_core), .Q(issue_token_tag[2]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_1_ ( .CN(n841), .D(n331), .CP(
        clk_core), .Q(issue_token_tag[1]) );
  DFKCNQD1BWP35P140 issue_token_tag_q_reg_0_ ( .CN(n841), .D(n330), .CP(
        clk_core), .Q(issue_token_tag[0]) );
  OAI22D0BWP35P140 U600 ( .A1(n682), .A2(window_token_tag[28]), .B1(n680), 
        .B2(window_token_tag[24]), .ZN(n561) );
  AOI221D0BWP35P140 U605 ( .A1(n682), .A2(window_token_tag[28]), .B1(
        window_token_tag[24]), .B2(n680), .C(n561), .ZN(n562) );
  AOI221D0BWP35P140 U608 ( .A1(n716), .A2(window_token_tag[39]), .B1(
        window_token_tag[41]), .B2(n708), .C(n545), .ZN(n546) );
  ND2D0BWP35P140 U612 ( .A1(window_head_channel[110]), .A2(
        window_head_channel[109]), .ZN(n587) );
  ND2D0BWP35P140 U615 ( .A1(window_head_channel[144]), .A2(
        window_head_channel[145]), .ZN(n572) );
  NR2D0BWP35P140 U616 ( .A1(window_bank_count[43]), .A2(window_bank_count[41]), 
        .ZN(n494) );
  OR2D0BWP35P140 U617 ( .A1(window_bank_count[68]), .A2(window_bank_count[66]), 
        .Z(n523) );
  NR3D0BWP35P140 U618 ( .A1(window_bank_count[73]), .A2(window_bank_count[79]), 
        .A3(window_bank_count[77]), .ZN(n468) );
  NR2D0BWP35P140 U619 ( .A1(window_bank_count[35]), .A2(window_bank_count[34]), 
        .ZN(n501) );
  NR3D0BWP35P140 U620 ( .A1(n476), .A2(window_bank_count[126]), .A3(
        window_bank_count[60]), .ZN(n788) );
  NR2D0BWP35P140 U621 ( .A1(window_bank_count[50]), .A2(window_bank_count[49]), 
        .ZN(n509) );
  OR2D0BWP35P140 U622 ( .A1(window_bank_count[23]), .A2(window_bank_count[22]), 
        .Z(n478) );
  NR2D0BWP35P140 U623 ( .A1(window_bank_count[95]), .A2(window_bank_count[93]), 
        .ZN(n528) );
  NR2D0BWP35P140 U624 ( .A1(n614), .A2(n613), .ZN(n602) );
  ND2D0BWP35P140 U625 ( .A1(n538), .A2(n537), .ZN(n734) );
  ND3D0BWP35P140 U626 ( .A1(n541), .A2(n540), .A3(n539), .ZN(n631) );
  OR2D0BWP35P140 U627 ( .A1(window_bank_count[51]), .A2(window_bank_count[48]), 
        .Z(n508) );
  ND3D0BWP35P140 U631 ( .A1(n473), .A2(n793), .A3(n792), .ZN(n474) );
  ND2D0BWP35P140 U633 ( .A1(n491), .A2(n490), .ZN(n667) );
  ND3D0BWP35P140 U634 ( .A1(n602), .A2(n809), .A3(n813), .ZN(n597) );
  NR3D0BWP35P140 U641 ( .A1(n474), .A2(window_bank_count[64]), .A3(
        window_bank_count[5]), .ZN(n807) );
  OAI211D0BWP35P140 U650 ( .A1(n598), .A2(n597), .B(n596), .C(n595), .ZN(n600)
         );
  AO21D0BWP35P140 U651 ( .A1(pair_valid), .A2(n600), .B(fault_q), .Z(
        protocol_error) );
  AN2D0BWP35P140 U656 ( .A1(pair_ready), .A2(pair_valid), .Z(pair_accept) );
  TIEHBWP35P140 U661 ( .Z(n841) );
  INVD1BWP35P140 U673 ( .I(n841), .ZN(issue_source_channel[12]) );
  INVD1BWP35P140 U687 ( .I(n841), .ZN(issue_source_channel[25]) );
  INVD1BWP35P140 U691 ( .I(n841), .ZN(issue_source_channel[36]) );
  INVD1BWP35P140 U699 ( .I(n841), .ZN(issue_source_channel[37]) );
  INVD1BWP35P140 U716 ( .I(n841), .ZN(issue_source_channel[50]) );
  INVD1BWP35P140 U723 ( .I(n841), .ZN(issue_source_channel[60]) );
  INVD1BWP35P140 U739 ( .I(n841), .ZN(issue_source_channel[62]) );
  INVD1BWP35P140 U756 ( .I(n841), .ZN(issue_source_channel[73]) );
  INVD1BWP35P140 U789 ( .I(n841), .ZN(issue_source_channel[74]) );
  INVD1BWP35P140 U790 ( .I(n841), .ZN(issue_source_channel[84]) );
  INVD1BWP35P140 U807 ( .I(n841), .ZN(issue_source_channel[85]) );
  INVD1BWP35P140 U811 ( .I(n841), .ZN(issue_source_channel[86]) );
endmodule

