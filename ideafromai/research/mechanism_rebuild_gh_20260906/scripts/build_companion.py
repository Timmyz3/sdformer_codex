"""Export the reported CPU opportunity data, without introducing new experiments."""
from pathlib import Path
import hashlib
import json

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'records/threshold_packet_sample0_screen.json'
OUT = BASE / 'opportunity-data.xlsx'


def main():
    data = json.loads(SOURCE.read_text())
    wb = Workbook()
    notes = wb.active
    notes.title = '说明与来源'
    notes.append(['项目', '内容'])
    note_rows = [
        ['证据身份', '2026-09-06；ep34 同 SHA 参数；sample0；两个预声明 FC1 层；CPU float64 重建。'],
        ['边界', '非 RTL 周期、非 PPA、非全域浮点等价证明；32 位只用于容量示例。'],
        ['输入来源', str(SOURCE)],
        ['输入 SHA256', hashlib.sha256(SOURCE.read_bytes()).hexdigest()],
        ['checkpoint SHA256', data['checkpoint_sha256']],
        ['数据范围', '全部 2 层 × 12 组预声明 B/K，保留原始计数与派生指标；不嵌入大型二值捕获。'],
        ['百分比含义', '表中 fraction 列保存 0–1 比值；分母为本层首遍完整 FC1 活跃标量加项。'],
        ['完整宽值 MB', 'N_includes_T × H × 4 / 1,000,000；U 位宽假设为 32。'],
        ['逐 h 读源 MB', 'replay_PSN_terms_by_h × C / (8 × T) / 1,000,000；无跨 h 共享，不是外存下界。'],
        ['裸编码 MB', 'bare_payload_bytes_by_assumed_U_width[32] / 1,000,000；未计源、临时缓冲、宏对齐、队列及端口。'],
        ['导出方式', '数值从已保存 JSON 导出；派生值使用上列公式；不依赖 Excel 重算或宏。'],
        ['PPA_ADMISSION', 0],
        ['RTL_SPEEDUP_ADMISSION', 0],
    ]
    for row in note_rows:
        notes.append(row)
    raw = wb.create_sheet('全部24组')
    headers = ['module', 'N_includes_T', 'P', 'C', 'H', 'T', 'B', 'K',
               'packets', 'failed_packets', 'packet_failure_fraction',
               'full_FC1_active_scalar_terms', 'replay_FC1_terms_by_h',
               'replay_FC1_fraction_by_h', 'replay_FC1_terms_fixed96',
               'replay_FC1_fraction_fixed96', 'full_PSN_scalar_terms',
               'replay_PSN_terms_by_h', 'certified_patch_bits',
               'bare_payload_bytes_U32_assumption', 'bare_payload_bytes_U64_assumption',
               'bare_payload_U32_MB', 'full_U32_MB', 'source_read_by_h_MB']
    raw.append(headers)
    expected = []
    for layer in data['layers']:
        for row in layer['rows']:
            values = [layer[k] for k in headers[:6]] + [row[k] for k in headers[6:19]] + [
                row['bare_payload_bytes_by_assumed_U_width']['32'],
                row['bare_payload_bytes_by_assumed_U_width']['64'],
                row['bare_payload_bytes_by_assumed_U_width']['32'] / 1e6,
                layer['N_includes_T'] * layer['H'] * 4 / 1e6,
                row['replay_PSN_terms_by_h'] * layer['C'] / (8 * layer['T']) / 1e6,
            ]
            assert abs(values[10] - values[9] / values[8]) < 1e-12
            assert abs(values[13] - values[12] / values[11]) < 1e-12
            assert abs(values[15] - values[14] / values[11]) < 1e-12
            raw.append(values)
            expected.append(values)
    for ws in wb:
        ws.freeze_panes = 'A2'
        for cell in ws[1]:
            cell.font = Font(name='Calibri', bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='173F49')
            cell.alignment = Alignment(wrap_text=True, vertical='center')
        ws.row_dimensions[1].height = 42
    notes.column_dimensions['A'].width = 25
    notes.column_dimensions['B'].width = 112
    for row in notes.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical='top')
        notes.row_dimensions[row[0].row].height = 35
    raw.auto_filter.ref = raw.dimensions
    raw.column_dimensions['A'].width = 62
    for col in range(2, len(headers) + 1):
        raw.column_dimensions[raw.cell(1, col).column_letter].width = 23
    for row in raw.iter_rows(min_row=2):
        for col in (11, 14, 16):
            row[col - 1].number_format = '0.0000%'
        for col in (22, 23, 24):
            row[col - 1].number_format = '0.000'
    wb.save(OUT)
    checked = load_workbook(OUT, read_only=True, data_only=True)
    observed = list(checked['全部24组'].values)
    assert observed[0] == tuple(headers)
    assert len(observed) == 25
    for original, restored in zip(expected, observed[1:], strict=True):
        for a, b in zip(original, restored, strict=True):
            if isinstance(a, (int, float)):
                assert abs(a - b) <= 1e-12 * max(1, abs(a))
            else:
                assert a == b
    checked.close()
    result = {'status': 'PASS_EXPORTED_VALUES_AND_RATIO_CHECKS',
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'workbook_sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
              'data_rows': len(expected), 'columns': len(headers),
              'scope': 'Companion data for existing report chart; no new experiment, formulas, macros or external workbook connections.',
              'visual_qa': 'NOT_PERFORMED'}
    (BASE / 'records/opportunity_workbook_qa.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
