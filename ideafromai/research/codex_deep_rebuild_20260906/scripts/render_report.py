#!/usr/bin/env python3.12
"""Render the canonical Markdown research report to a portable Chinese PDF."""
import sys
from pathlib import Path
sys.path.insert(0,'/tmp/codex_c1c2_report_vendor')
import html,re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
from reportlab.graphics.shapes import Drawing,Rect,Line,Polygon,String

ROOT=Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont('YaHei','/usr/share/fonts/chinese/msyh.ttc',subfontIndex=0))
bold=Path('/usr/share/fonts/chinese/msyhbd.ttc')
pdfmetrics.registerFont(TTFont('YaHeiBold',str(bold if bold.exists() else '/usr/share/fonts/chinese/msyh.ttc'),subfontIndex=0))
pdfmetrics.registerFontFamily('YaHei',normal='YaHei',bold='YaHeiBold',italic='YaHei',boldItalic='YaHeiBold')
NAVY=colors.HexColor('#15304B');BLUE=colors.HexColor('#146B91');GRAY=colors.HexColor('#586473')
body=ParagraphStyle('body',fontName='YaHei',fontSize=10.1,leading=16.1,wordWrap='CJK',textColor=NAVY,spaceAfter=9)
title=ParagraphStyle('title',parent=body,fontName='YaHeiBold',fontSize=20,leading=28,spaceAfter=17)
heading=ParagraphStyle('heading',parent=title,fontSize=17,leading=24,spaceAfter=16)
cell=ParagraphStyle('cell',parent=body,fontSize=8.65,leading=13.5,spaceAfter=0)
cellhead=ParagraphStyle('cellhead',parent=cell,fontName='YaHeiBold',textColor=colors.white)

def inline(text):
    x=html.escape(text)
    x=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#146B91"><u>\1</u></link>',x)
    x=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',x)
    x=re.sub(r'`([^`]+)`',r'<font color="#146B91">\1</font>',x)
    return x

def diagram(kind):
    d=Drawing(503,108)
    def box(x,y,w,h,lines,fill='#EFF5F8'):
        d.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=colors.HexColor(fill),strokeColor=BLUE,strokeWidth=.65))
        for i,t in enumerate(lines):d.add(String(x+w/2,y+h/2+(len(lines)-1)*6-i*12,t,fontName='YaHei',fontSize=8.5,textAnchor='middle',fillColor=NAVY))
    def arrow(x1,y1,x2,y2):
        d.add(Line(x1,y1,x2,y2,strokeColor=BLUE,strokeWidth=.8))
        if x2>x1:d.add(Polygon([x2,y2,x2-4,y2+2,x2-4,y2-2],fillColor=BLUE,strokeColor=BLUE))
    if kind=='C1':
        box(0,35,98,40,['二值支持 / 子集森林','深度优先穿线'])
        box(120,35,76,40,['查 2/4 父槽'])
        box(223,65,117,35,['命中：父结果 + 残差'])
        box(223,8,117,35,['缺失：从零发原支持'])
        box(373,35,128,40,['当前行唯一提交','释放 / FIFO 入槽'])
        arrow(98,55,120,55);arrow(196,55,223,82);arrow(196,55,223,26)
        arrow(340,82,373,55);arrow(340,26,373,55)
    else:
        box(0,36,104,43,['已接受的八 bank','权重返回'])
        box(134,67,132,34,['两槽 INT8 权重捕获'])
        box(134,7,132,34,['普通消费者更新'])
        box(302,35,113,43,['八输入归约树','仅空闲输入 0/1 注入'])
        box(439,36,62,43,['Acc24','原上下文'])
        arrow(104,57,134,84);arrow(104,57,134,24)
        arrow(266,84,302,63);arrow(266,24,302,47);arrow(415,57,439,57)
    return d

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#D8E2E8'));canvas.line(46,43,A4[0]-46,43)
    canvas.setFont('YaHei',8);canvas.setFillColor(GRAY)
    canvas.drawString(46,29,'C1/C2 机制重构研究  ·  2026-09-06  ·  CPU 筛查证据')
    canvas.drawRightString(A4[0]-46,29,str(doc.page));canvas.restoreState()

def main():
    pages=(ROOT/'report-source.md').read_text().split('<!-- PAGE -->');story=[]
    for page_no,page in enumerate(pages,1):
        lines=page.strip().splitlines();i=0
        while i<len(lines):
            line=lines[i].strip()
            if not line:i+=1;continue
            if line.startswith('# '):story.append(Paragraph(inline(line[2:]),title if page_no==1 else heading));i+=1;continue
            if line.startswith('[[DIAGRAM:'):
                story.extend([diagram(line.split(':')[1].split(']')[0]),Spacer(1,11)]);i+=1;continue
            if line.startswith('|'):
                rows=[]
                while i<len(lines) and lines[i].strip().startswith('|'):
                    bits=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                    if not all(re.fullmatch(r'[:\-\s]+',v) for v in bits):rows.append(bits)
                    i+=1
                n=len(rows[0]);widths={1:[58,160,213,72],2:[135,166,202],3:[82,48,110,125,138],5:[242,83,83,95],8:[134,213,156]}.get(page_no,[503/n]*n)
                t=Table([[Paragraph(inline(v),cellhead if r==0 else cell) for v in row] for r,row in enumerate(rows)],colWidths=widths,repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F1F5F8'),colors.white]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#D8E2E8'))]))
                story.extend([t,Spacer(1,12)]);continue
            para=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('# ','|','[[DIAGRAM:')):
                para.append(lines[i].strip());i+=1
            story.append(Paragraph(inline(' '.join(para)),body))
        if page_no<len(pages):story.append(PageBreak())
    out=ROOT/'C1C2机制重构研究与筛查_20260906.pdf'
    doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=43,bottomMargin=59,title='C1/C2 机制重构：文献迁移与同账本筛查',author='工程研究记录',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(out)
if __name__=='__main__':main()
