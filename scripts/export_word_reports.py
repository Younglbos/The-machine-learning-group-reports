"""Export existing verified report content as editable DOCX without rerunning experiments.

Use the Documents workspace runtime's Python (python-docx, lxml and Pillow).
Wide scientific figures are reflowed as their unchanged subplot panels so their
axis labels remain readable in a portrait document. No plotted values are edited.
"""

import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile
import itertools
import numpy as np

from lxml import html
from lxml import etree
from PIL import Image
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches,Pt,RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'
WORD_NAMES={'course_report':'多视图聚类课程设计_实验报告.docx',
            'literature_review':'多视图聚类课程设计_文献综述.docx'}
TITLES={'course_report':'不完整视图与多视图流聚类实验报告',
        'literature_review':'不完整多视图与流式多视图聚类文献综述'}
PANELS={
    'course_a_missingness.png':(1,3,75,['随机缺失','单视图集中缺失','类别相关缺失']),
    'course_a_geometry_neighbors.png':(1,2,90,['第一视图缺失行的邻域保持','第二视图缺失行的邻域保持']),
    'course_a_geometry_pca.png':(1,3,225,['真实完整原始特征','观测特征与解码补全','完成后的共识潜表示及类别图例']),
    'course_a_ablations.png':(1,3,70,['ACC','NMI','ARI']),
    'course_a_robustness.png':(1,2,70,['ACC','NMI']),
    'course_b_quality.png':(2,3,90,['无漂移 ACC','无漂移 NMI','无漂移 ARI','视图加噪 ACC','视图加噪 NMI','视图加噪 ARI']),
    'course_b_retention.png':(1,2,70,['无漂移历史查询','视图加噪历史查询']),
    'course_b_label_changes.png':(1,2,70,['无漂移标签变化','视图加噪标签变化']),
    'course_b_time.png':(1,2,70,['无漂移累计时间','视图加噪累计时间']),
    'course_b_state.png':(1,2,70,['无漂移摘要状态','视图加噪摘要状态']),
}


def font(run,size=None,bold=None,east_asia='宋体'):
    run.font.name='Times New Roman'
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),east_asia)
    if size is not None: run.font.size=Pt(size)
    if bold is not None: run.bold=bold
    run.font.color.rgb=RGBColor(0,0,0)


def setup_document():
    doc=Document()
    section=doc.sections[0]
    section.page_width=Inches(8.5)
    section.page_height=Inches(11)
    section.top_margin=Inches(.72)
    section.bottom_margin=Inches(.72)
    section.left_margin=Inches(.78)
    section.right_margin=Inches(.78)
    section.footer_distance=Inches(.3)
    normal=doc.styles['Normal']
    normal.font.name='Times New Roman'
    normal.font.size=Pt(11.5)
    normal._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'宋体')
    normal.paragraph_format.line_spacing=1.35
    normal.paragraph_format.space_after=Pt(6)
    normal.paragraph_format.widow_control=True
    for style_name,size in [('Title',20),('Subtitle',12),('Heading 1',15),('Heading 2',12.5)]:
        style=doc.styles[style_name]
        style.font.name='Times New Roman'
        style.font.size=Pt(size)
        style.font.color.rgb=RGBColor(0,0,0)
        style.font.bold=style_name!='Subtitle'
        style.font.italic=False
        style._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'黑体')
        style.paragraph_format.space_before=Pt(12 if style_name.startswith('Heading') else 0)
        style.paragraph_format.space_after=Pt(8)
        style.paragraph_format.keep_with_next=True
        borders=style._element.find('.//'+qn('w:pBdr'))
        if borders is not None: borders.getparent().remove(borders)
    doc.styles['Caption'].font.size=Pt(10)
    doc.styles['Caption'].font.color.rgb=RGBColor(0,0,0)
    doc.styles['Caption'].paragraph_format.space_after=Pt(5)
    doc.styles['Caption'].paragraph_format.line_spacing=1.15
    footer=section.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple')
    field.set(qn('w:instr'),'PAGE')
    run=OxmlElement('w:r')
    text=OxmlElement('w:t')
    text.text='1'
    run.append(text)
    field.append(run)
    footer._p.append(field)
    doc.core_properties.author=''
    doc.core_properties.last_modified_by=''
    doc.core_properties.subject='多视图聚类课程设计'
    doc.core_properties.language='zh-CN'
    return doc


def hyperlink(paragraph,label,url):
    if url=='literature_review.html': url=WORD_NAMES['literature_review']
    rel_id=paragraph.part.relate_to(url,RELATIONSHIP_TYPE.HYPERLINK,is_external=True)
    link=OxmlElement('w:hyperlink')
    link.set(qn('r:id'),rel_id)
    run=OxmlElement('w:r')
    props=OxmlElement('w:rPr')
    color=OxmlElement('w:color'); color.set(qn('w:val'),'174B73'); props.append(color)
    fonts=OxmlElement('w:rFonts'); fonts.set(qn('w:ascii'),'Times New Roman'); fonts.set(qn('w:hAnsi'),'Times New Roman'); fonts.set(qn('w:eastAsia'),'宋体'); props.append(fonts)
    run.append(props)
    text=OxmlElement('w:t'); text.text=label; run.append(text)
    link.append(run); paragraph._p.append(link)


def inline(paragraph,node,bold=False,subscript=False):
    if node.text:
        run=paragraph.add_run(node.text); font(run,bold=bold); run.font.subscript=subscript
    for child in node:
        if child.tag=='a': hyperlink(paragraph,child.text_content(),child.get('href',''))
        elif child.tag=='br': paragraph.add_run().add_break()
        else: inline(paragraph,child,bold or child.tag in {'b','strong'},subscript or child.tag=='sub')
        if child.tail:
            run=paragraph.add_run(child.tail); font(run,bold=bold); run.font.subscript=subscript


def equation(paragraph):
    math=OxmlElement('m:oMath')
    def mrun(text):
        run=OxmlElement('m:r'); value=OxmlElement('m:t'); value.text=text; run.append(value); return run
    math.append(mrun('L = '))
    for i,(coefficient,label) in enumerate([('0.1','reconstruction'),('10','instance'),('1','MMD'),('0.1','dual')]):
        math.append(mrun((' + ' if i else '')+(coefficient+' ' if coefficient!='1' else '')))
        sub=OxmlElement('m:sSub')
        base=OxmlElement('m:e'); base.append(mrun('L')); sub.append(base)
        index=OxmlElement('m:sub'); index.append(mrun(label)); sub.append(index)
        math.append(sub)
    paragraph._p.append(math)


def add_paragraph(doc,node):
    text=node.text_content().strip()
    if not text: return
    if text.startswith('姓名、学号、班级与指导教师'): return
    # Omit chat workflow notes from the academic deliverable; preserve scientific limitations.
    text=text.replace('按用户安排，PPT在实验、报告与代码核验后再制作；本报告不把尚未制作的PPT列为已交付。','')
    paragraph=doc.add_paragraph()
    paragraph.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    if 'L = 0.1 L' in text:
        prefix,suffix=text.split('L = ',1)
        suffix=suffix.split('Ldual',1)[1]
        run=paragraph.add_run(prefix); font(run)
        equation(paragraph)
        run=paragraph.add_run(suffix); font(run)
    elif text!=node.text_content().strip():
        run=paragraph.add_run(text); font(run)
    else: inline(paragraph,node)
    return paragraph


def heading_text(text):
    # Keep subject and hierarchy; headings use plain words/numbers without punctuation.
    text=re.sub(r'^(\d+)\.(\d+)\s*','',text)
    text=re.sub(r'^(\d+)\.\s*',r'\1 ',text)
    text=re.sub(r'[：:、，,。/—–（）()]',' ',text)
    return re.sub(r'\s+',' ',text).strip()


def add_table(doc,node):
    caption=node.find('caption')
    if caption is not None and caption.text_content().strip():
        p=doc.add_paragraph(caption.text_content().strip(),'Caption'); p.paragraph_format.keep_with_next=True
    rows=node.xpath('./thead/tr|./tbody/tr|./tr')
    if not rows: return
    cells=[row.xpath('./th|./td') for row in rows]
    columns=max(map(len,cells))
    table=doc.add_table(rows=0,cols=columns)
    table.alignment=WD_TABLE_ALIGNMENT.CENTER
    table.autofit=False
    total=6.94
    if columns==2: widths=[1.7,total-1.7]
    elif columns==3: widths=[1.25,2.25,total-3.5]
    elif columns==5 and cells[0][0].text_content()=='数据集': widths=[1.65,.65,.65,.65,3.34]
    elif columns==5 and cells[0][0].text_content()=='视图': widths=[.5,1.15,1.65,1.65,1.99]
    elif columns==6: widths=[1.25,.8,1.29,1.2,1.2,1.2]
    elif columns==7: widths=[.6,1.25,1.02,1.02,1.02,1.02,1.01]
    else: widths=[total/columns]*columns
    for col,width in zip(table.columns,widths): col.width=Inches(width)
    for row_index,source in enumerate(cells):
        row=table.add_row()
        props=row._tr.get_or_add_trPr(); props.append(OxmlElement('w:cantSplit'))
        if row_index==0:
            header=OxmlElement('w:tblHeader'); header.set(qn('w:val'),'true'); props.append(header)
        for index,cell in enumerate(row.cells):
            cell.width=Inches(widths[index])
            cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=cell._tc.get_or_add_tcPr()
            margins=OxmlElement('w:tcMar')
            for side,value in [('top',80),('bottom',80),('left',70),('right',70)]:
                part=OxmlElement('w:'+side); part.set(qn('w:w'),str(value)); part.set(qn('w:type'),'dxa'); margins.append(part)
            tcpr.append(margins)
            if row_index==0:
                shade=OxmlElement('w:shd'); shade.set(qn('w:fill'),'F0F0F0'); tcpr.append(shade)
            paragraph=cell.paragraphs[0]
            paragraph.paragraph_format.space_before=Pt(0)
            paragraph.paragraph_format.space_after=Pt(0)
            paragraph.paragraph_format.line_spacing=1.13
            paragraph.paragraph_format.keep_with_next=row_index==0
            if index<len(source): inline(paragraph,source[index])
            for run in paragraph.runs: font(run,10,bold=row_index==0,east_asia='黑体' if row_index==0 else '宋体')
    borders=OxmlElement('w:tblBorders')
    for side in ['top','bottom','left','right','insideH','insideV']:
        border=OxmlElement('w:'+side); border.set(qn('w:val'),'single'); border.set(qn('w:sz'),'4'); border.set(qn('w:color'),'BBBBBB'); borders.append(border)
    table._tbl.tblPr.append(borders)
    doc.add_paragraph().paragraph_format.space_after=Pt(2)


def reflow_figure(source,destination):
    image=Image.open(source).convert('RGB')
    width,height=image.size
    rows,cols,top,labels=PANELS[source.name]
    outputs=[]
    black=np.asarray(image.convert('L'))<100
    spine_rows=[list(group)[-1] for matches,group in itertools.groupby(range(height),lambda y:black[y].sum()>width*.55) if matches]
    bottom=spine_rows[-1]
    spans=[(group[0],group[-1]+1) for group in [list(g) for matches,g in itertools.groupby(range(width),lambda x:black[bottom,x]) if matches] if len(group)>width*.12]
    if len(spans)!=cols: raise ValueError(f'Cannot separate complete subplot axes: {source}')
    boundaries=[0]+[span[1]+8 for span in spans[:-1]]+[width]
    for i,label in enumerate(labels):
        r,c=divmod(i,cols)
        left,right=boundaries[c:c+2]
        if source.name=='course_a_geometry_pca.png': top=232
        if rows==1: y0,y1=top,height
        else: y0,y1=[(90,535),(535,height)][r]
        crop=image.crop((left,y0,right,y1))
        output=destination/f'{source.stem}_{i+1}.png'
        crop.save(output)
        outputs.append((output,label))
    return outputs


def add_figure(doc,node,cache,figure_index):
    element=node.find('img')
    source=OUT/element.get('src')
    caption=node.find('figcaption').text_content()
    p=doc.add_paragraph(caption,'Caption'); p.paragraph_format.keep_with_next=True
    panels=reflow_figure(source,cache)
    for i,(path,label) in enumerate(panels):
        p=doc.add_paragraph()
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after=Pt(2)
        p.paragraph_format.keep_with_next=True
        shape=p.add_run().add_picture(str(path),width=Inches(5.0))
        shape._inline.docPr.set('descr',f'{caption} 子图{chr(97+i)} {label}')
        p=doc.add_paragraph(f'图{figure_index}{chr(97+i)} {label}','Caption')
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next=False
    return len(panels)


def node_data(node):
    return {'tag':node.tag,'attributes':dict(node.attrib),'text':node.text,'tail':node.tail,
            'children':[node_data(child) for child in node]}


def data_node(data):
    node=etree.Element(data['tag'],attrib=data['attributes'])
    node.text,node.tail=data['text'],data['tail']
    for child in data['children']: node.append(data_node(child))
    return node


def export_one(kind,cache,sources_only=False):
    source=OUT/'sources'/(kind+'.json')
    source.parent.mkdir(exist_ok=True)
    intermediate=ROOT/'.cache/word_sources'/(kind+'.html')
    legacy=OUT/(kind+'.html')
    candidate=intermediate if intermediate.is_file() else legacy
    if candidate.is_file() and not sources_only:
        root=html.fromstring(candidate.read_bytes())
        source.write_text(json.dumps(node_data(root),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    else:
        root=html.fromstring(etree.tostring(data_node(json.loads(source.read_text(encoding='utf-8')))))
    doc=setup_document()
    title=doc.add_paragraph(TITLES[kind],'Title')
    title.alignment=WD_ALIGN_PARAGRAPH.CENTER
    subtitle=doc.add_paragraph('多视图聚类课程设计','Subtitle')
    subtitle.alignment=WD_ALIGN_PARAGRAPH.CENTER
    meta=doc.add_paragraph('实验与文献核验日期 2026年11月24日    Word版整理日期 2026年11月25日')
    meta.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for run in meta.runs: font(run,9.5)
    if kind=='course_report': blocks=root.xpath('//main/section/*')
    else: blocks=root.xpath('//body/*')
    figures=panels=0
    for node in blocks:
        tag=node.tag
        if tag in {'h1','meta','title','style','header'}: continue
        if tag in {'h2','h3'}:
            text=node.text_content().strip()
            style='Heading 2' if re.match(r'^\d+\.\d+',text) or tag=='h3' else 'Heading 1'
            doc.add_paragraph(heading_text(text),style)
        elif tag=='p': add_paragraph(doc,node)
        elif tag=='table': add_table(doc,node)
        elif tag=='div' and node.get('class')=='table':
            for table in node.findall('table'): add_table(doc,table)
        elif tag=='figure': figures+=1; panels+=add_figure(doc,node,cache,figures)
        elif tag=='pre':
            for line in node.text_content().strip().splitlines():
                paragraph=doc.add_paragraph()
                paragraph.paragraph_format.line_spacing=1.05
                paragraph.paragraph_format.space_after=Pt(3)
                paragraph.paragraph_format.left_indent=Inches(.1)
                run=paragraph.add_run(line); font(run,9)
        elif tag in {'ol','ul'}:
            for i,item in enumerate(node.findall('li'),1):
                p=doc.add_paragraph()
                p.paragraph_format.left_indent=Inches(.26)
                p.paragraph_format.first_line_indent=Inches(-.26)
                p.add_run(f'[{i}] ' if tag=='ol' else '• ')
                inline(p,item)
    doc.core_properties.title=TITLES[kind]
    target=OUT/WORD_NAMES[kind]
    doc.save(target)
    with ZipFile(target) as zipped:
        if not {'word/document.xml','word/styles.xml','[Content_Types].xml'}.issubset(zipped.namelist()): raise ValueError('Invalid DOCX')
        media=[x for x in zipped.namelist() if x.startswith('word/media/')]
        if len(media)!=panels: raise ValueError('Figure embedding mismatch')
    return {'file':target.relative_to(ROOT).as_posix(),'source':source.relative_to(ROOT).as_posix(),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'tables':len(doc.tables),'source_figures':figures,'embedded_panels':panels,
        'paragraphs':len(doc.paragraphs),'render_qa':'pending'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',default='.cache/word_reports/panels')
    parser.add_argument('--sources-only',action='store_true',help='Rebuild only from published JSON document sources')
    args=parser.parse_args()
    cache=ROOT/args.cache_dir
    cache.mkdir(parents=True,exist_ok=True)
    rows=[export_one(kind,cache,args.sources_only) for kind in ['course_report','literature_review']]
    (OUT/'word_export_manifest.json').write_text(json.dumps({'documents':rows,'experimental_results_changed':False,
        'format':'Office Open XML .docx','ppt_status':'deferred_by_user'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
