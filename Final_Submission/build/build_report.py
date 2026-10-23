"""Builds the final project report from the faculty template.

The cover page and its logo are kept from the template exactly; only the article
title is filled in. The contents page is rebuilt with real page numbers, and the
body sections are appended in a new, page-numbered section.

Usage: python build_report.py <template.docx> <figures_dir> <output.docx>
Needs LibreOffice and pdftotext for the page-number pass.
"""
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from report_content import CONTENT, TITLE
from scripts_content import MEMBER_ORDER, MEMBERS

TEXT_WIDTH = 9360  # US Letter with 1" margins, in twips
SOFFICE = "/home/shafreed/.claude/skills/synced/f7d7525f-fac9-4ce7-b48f-0dbec9262a24_7916b980-97b8-4f86-91a9-c6e63cf9e18d/docx/scripts/office/soffice.py"
FONT = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
NAVY = "1F3A5F"


# ---------------------------------------------------------------- runs ----
def run(text, b=False, i=False, sz=24, font=FONT, color=None, sup=False):
    props = font
    if b: props += "<w:b/><w:bCs/>"
    if i: props += "<w:i/><w:iCs/>"
    if color: props += f'<w:color w:val="{color}"/>'
    if sup: props += '<w:vertAlign w:val="superscript"/>'
    props += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
    return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def rich(text, sz=24, b=False, i=False):
    """Inline markup: **bold**, *italic*, `code`, ^{superscript}."""
    out = []
    for part in re.split(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|\^\{.+?\})", text):
        if not part:
            continue
        if part.startswith("**"):
            out.append(run(part[2:-2], b=True, i=i, sz=sz))
        elif part.startswith("^{"):
            out.append(run(part[2:-1], b=b, i=i, sz=sz, sup=True))
        elif part.startswith("`"):
            out.append(run(part[1:-1], b=b, sz=sz - 2,
                           font='<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>'))
        elif part.startswith("*"):
            out.append(run(part[1:-1], b=b, i=True, sz=sz))
        else:
            out.append(run(part, b=b, i=i, sz=sz))
    return "".join(out)


def para(runs, jc="both", after=120, before=0, line=360, style=None, extra=""):
    st = f'<w:pStyle w:val="{style}"/>' if style else ""
    return (f'<w:p><w:pPr>{st}{extra}<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" '
            f'w:lineRule="auto"/><w:jc w:val="{jc}"/></w:pPr>{runs}</w:p>')


# --------------------------------------------------------------- blocks ---
class Doc:
    def __init__(self, figures_dir):
        self.figures_dir = Path(figures_dir)
        self.images = []          # (rId, filename)
        self.lists = 0            # numbered lists need their own w:num
        self.fig_no = 0
        self.tab_no = 0
        self.docpr = 100

    def h1(self, text):
        return f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>{run(text, b=True, sz=36)}</w:p>'

    def h2(self, text):
        return f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr>{run(text, b=True, sz=28)}</w:p>'

    def p(self, text):
        return para(rich(text))

    def plead(self, text):
        """A paragraph that introduces the next block (e.g. code) and stays on its page."""
        return para(rich(text), extra="<w:keepNext/><w:keepLines/>")

    def bullets(self, items, num_id=1):
        return "".join(
            f'<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num_id}"/></w:numPr>'
            f'<w:spacing w:after="80" w:line="336" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>{rich(t)}</w:p>'
            for t in items)

    def numbered(self, items):
        self.lists += 1
        return self.bullets(items, num_id=100 + self.lists)

    def eq(self, text):
        return para(run(text, i=True, sz=24, font='<w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>'),
                    jc="center", before=60, after=180, line=276)

    def code(self, lines):
        font = '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:cs="Courier New"/>'
        out = []
        for k, ln in enumerate(lines):
            after = 160 if k == len(lines) - 1 else 0
            keep = "<w:keepNext/><w:keepLines/>" if k < len(lines) - 1 else "<w:keepLines/>"
            out.append(f'<w:p><w:pPr>{keep}<w:shd w:val="clear" w:color="auto" w:fill="F2F4F7"/>'
                       f'<w:spacing w:after="{after}" w:line="264" w:lineRule="auto"/><w:ind w:left="284"/></w:pPr>'
                       f'{run(ln, sz=19, font=font)}</w:p>')
        return "".join(out)

    def caption(self, text):
        return para(rich(text, sz=21, i=True), jc="center", before=60, after=240, line=276)

    def table(self, caption, header, rows, widths):
        self.tab_no += 1
        scale = TEXT_WIDTH / sum(widths)
        w = [round(x * scale) for x in widths]
        w[-1] += TEXT_WIDTH - sum(w)
        border = '<w:{0} w:val="single" w:sz="4" w:space="0" w:color="9AA5B4"/>'
        borders = "".join(border.format(s) for s in ("top", "left", "bottom", "right", "insideH", "insideV"))

        def cell(t, width, head=False):
            shd = f'<w:shd w:val="clear" w:color="auto" w:fill="{NAVY}"/>' if head else ""
            r = run(t, b=head, sz=21, color="FFFFFF" if head else None) if head else rich(t, sz=21)
            return (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shd}<w:vAlign w:val="center"/></w:tcPr>'
                    f'<w:p><w:pPr><w:spacing w:before="40" w:after="40" w:line="252" w:lineRule="auto"/></w:pPr>{r}</w:p></w:tc>')

        trs = [f'<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>{"".join(cell(t, w[k], True) for k, t in enumerate(header))}</w:tr>']
        for row in rows:
            trs.append(f'<w:tr><w:trPr><w:cantSplit/></w:trPr>{"".join(cell(t, w[k]) for k, t in enumerate(row))}</w:tr>')
        grid = "".join(f'<w:gridCol w:w="{x}"/>' for x in w)
        tbl = (f'<w:tbl><w:tblPr><w:tblW w:w="{TEXT_WIDTH}" w:type="dxa"/><w:tblBorders>{borders}</w:tblBorders>'
               f'<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar>'
               f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>')
        cap = para(rich(f"**Table {self.tab_no}:** {caption}", sz=21), jc="center", before=120, after=80, line=276,
                   extra="<w:keepNext/>")
        return cap + tbl + para("", after=120)

    def figure(self, filename, width_in, caption):
        self.fig_no += 1
        self.docpr += 1
        rid = f"rIdFig{self.fig_no}"
        self.images.append((rid, filename))
        with open(self.figures_dir / filename, "rb") as fh:
            head = fh.read(24)
        px_w, px_h = struct.unpack(">II", head[16:24])
        cx = int(width_in * 914400)
        cy = int(cx * px_h / px_w)
        drawing = (
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{self.docpr}" name="Figure {self.fig_no}" descr="{escape(caption)}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{self.docpr}" name="{filename}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
        img = para(drawing, jc="center", before=120, after=40, line=240, extra="<w:keepNext/>")
        return img + self.caption(f"**Figure {self.fig_no}:** {caption}")

    def refs(self, items):
        return "".join(
            f'<w:p><w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/><w:ind w:left="567" w:hanging="567"/>'
            f'<w:jc w:val="left"/></w:pPr>{run(f"[{k}]", sz=24)}<w:r><w:tab/></w:r>{rich(t)}</w:p>'
            for k, t in enumerate(items, 1))

    def render(self, block):
        kind, *args = block
        return getattr(self, kind)(*args)


# --------------------------------------------------------------- parts ----
HEADING_STYLES = f"""
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/>
<w:pPr><w:keepNext/><w:keepLines/><w:pageBreakBefore/><w:spacing w:before="0" w:after="240"/><w:outlineLvl w:val="0"/></w:pPr>
<w:rPr>{FONT}<w:b/><w:bCs/><w:color w:val="000000"/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:uiPriority w:val="9"/><w:unhideWhenUsed/><w:qFormat/>
<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="1"/></w:pPr>
<w:rPr>{FONT}<w:b/><w:bCs/><w:color w:val="000000"/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:style>
""".replace("\n", "")


def numbering_xml(n_lists):
    bullet = ('<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="hybridMultilevel"/>'
              '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/>'
              f'<w:pPr><w:ind w:left="567" w:hanging="283"/></w:pPr><w:rPr>{FONT}</w:rPr></w:lvl></w:abstractNum>')
    decimal = ('<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="hybridMultilevel"/>'
               '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/>'
               f'<w:pPr><w:ind w:left="567" w:hanging="340"/></w:pPr><w:rPr>{FONT}</w:rPr></w:lvl></w:abstractNum>')
    nums = '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'
    for k in range(1, n_lists + 1):
        nums += (f'<w:num w:numId="{100 + k}"><w:abstractNumId w:val="1"/>'
                 '<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f'{bullet}{decimal}{nums}</w:numbering>')


FOOTER_XML = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
              'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
              f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr>{FONT}<w:sz w:val="22"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
              f'<w:r><w:rPr>{FONT}<w:sz w:val="22"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
              f'<w:r><w:rPr>{FONT}<w:sz w:val="22"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
              f'<w:r><w:rPr>{FONT}<w:sz w:val="22"/></w:rPr><w:t>1</w:t></w:r>'
              f'<w:r><w:rPr>{FONT}<w:sz w:val="22"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')

SECT_BASE = ('<w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" '
             'w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>')


def toc(entries, pages):
    out = []
    for k, title in enumerate(entries):
        page = str(pages.get(title, "")) if pages else ""
        sect = ""
        if k == len(entries) - 1:  # end of section 1: cover + contents, no page numbers
            sect = f'<w:sectPr>{SECT_BASE}<w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>'
        out.append(
            f'<w:p><w:pPr><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{TEXT_WIDTH}"/></w:tabs>'
            f'<w:spacing w:after="160" w:line="276" w:lineRule="auto"/>{sect}</w:pPr>'
            f'{run(title, sz=28)}<w:r><w:rPr>{FONT}<w:sz w:val="28"/></w:rPr><w:tab/></w:r>{run(page, sz=28)}</w:p>')
    return "".join(out)


def _para_around(x, needle):
    i = x.index(needle)
    s = x.rfind("<w:p ", 0, i)
    return s, x.index("</w:p>", i) + len("</w:p>")


def fill_members(x):
    """Replace the template's four StudentN(Roll No.) lines with the team, adding a fifth line."""
    names = [f"{MEMBERS[k][0]} ({MEMBERS[k][1]})" for k in MEMBER_ORDER]
    for n in range(1, 5):
        s, e = _para_around(x, f"Student{n}(Roll No.)")
        para_xml = re.sub(r"<w:t(?: [^>]*)?>[^<]*Student%d\(Roll No\.\)</w:t>" % n,
                          f"<w:t>{escape(names[n - 1])}</w:t>", x[s:e])
        x = x[:s] + para_xml + x[e:]
    # The template positions line 4 with tabs rather than centring it, so lines 4 onward are rebuilt
    # from the centred third line (without Word's paragraph ids)
    s3, e3 = _para_around(x, escape(names[2]))
    s4, e4 = _para_around(x, escape(names[3]))
    template_para = re.sub(r' w14:(paraId|textId)="[^"]*"', "", x[s3:e3])
    rebuilt = "".join(template_para.replace(escape(names[2]), escape(nm)) for nm in names[3:])
    x = x[:s4] + rebuilt + x[e4:]
    extra_len = len(rebuilt)
    # Each added line replaces one blank line below the names, keeping the logo on page 1
    pos = s4 + extra_len
    for _ in names[4:]:
        nxt_end = x.index("</w:p>", pos) + len("</w:p>")
        if "<w:t" in x[pos:nxt_end] or "<w:drawing" in x[pos:nxt_end]:
            break
        x = x[:pos] + x[nxt_end:]
    return x


def build(template, figures_dir, out_path, pages=None):
    work = Path(tempfile.mkdtemp(prefix="report_pkg_", dir=Path(out_path).parent))
    with zipfile.ZipFile(template) as z:
        z.extractall(work)
    for link in [p for p in work.rglob("*") if p.is_symlink()]:
        link.unlink()

    doc = Doc(figures_dir)
    body_xml = "".join(doc.render(b) for b in CONTENT)
    headings = [b[1] for b in CONTENT if b[0] == "h1"]

    # --- document.xml: cover title, contents page, body, final section ---
    dpath = work / "word" / "document.xml"
    x = dpath.read_text(encoding="utf8")
    assert "<w:t>TITLE OF THE ARTICLE</w:t>" in x
    x = x.replace("<w:t>TITLE OF THE ARTICLE</w:t>", f"<w:t>{escape(TITLE)}</w:t>")
    x = fill_members(x)
    # The real title wraps to two lines; drop the blank paragraph after it so the logo stays on page 1
    t_end = x.index("</w:p>", x.index(escape(TITLE))) + len("</w:p>")
    nxt_end = x.index("</w:p>", t_end) + len("</w:p>")
    if "<w:t" not in x[t_end:nxt_end] and "<w:drawing" not in x[t_end:nxt_end]:
        x = x[:t_end] + x[nxt_end:]
    # Contents starts on its own page
    c = x.index("<w:t>Contents</w:t>")
    p_start = x.rfind("<w:p ", 0, c)
    ppr = x.index("<w:pPr>", p_start)
    x = x[:ppr + 7] + "<w:pageBreakBefore/>" + x[ppr + 7:]
    # Empty paragraphs between the logo and "Contents" would spill onto a blank page 2
    logo_end = x.index("</w:p>", x.index("<w:drawing")) + len("</w:p>")
    between = x[logo_end:p_start]
    assert "<w:t" not in between and "<w:drawing" not in between
    x = x[:logo_end] + x[p_start:]
    # Replace the template's contents entries (from "Introduction……" to the end of the body)
    first = x.index("Introduction", x.index("Page No."))
    cut = x.rfind("<w:p ", 0, first)
    final_sect = (f'<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter1"/>{SECT_BASE}'
                  '<w:pgNumType w:start="1"/><w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>')
    x = x[:cut] + toc(headings, pages) + body_xml + final_sect + "</w:body></w:document>"
    dpath.write_text(x, encoding="utf8")

    # --- styles, numbering, footer, images, relationships, content types ---
    spath = work / "word" / "styles.xml"
    s = spath.read_text(encoding="utf8")
    spath.write_text(s.replace("</w:styles>", HEADING_STYLES + "</w:styles>"), encoding="utf8")
    (work / "word" / "numbering.xml").write_text(numbering_xml(doc.lists), encoding="utf8")
    (work / "word" / "footer1.xml").write_text(FOOTER_XML, encoding="utf8")
    for _rid, fname in doc.images:
        shutil.copy(Path(figures_dir) / fname, work / "word" / "media" / fname)

    rpath = work / "word" / "_rels" / "document.xml.rels"
    rels = rpath.read_text(encoding="utf8")
    R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    new_rels = (f'<Relationship Id="rIdFooter1" Type="{R}/footer" Target="footer1.xml"/>'
                f'<Relationship Id="rIdNumbering" Type="{R}/numbering" Target="numbering.xml"/>')
    new_rels += "".join(f'<Relationship Id="{rid}" Type="{R}/image" Target="media/{f}"/>' for rid, f in doc.images)
    rpath.write_text(rels.replace("</Relationships>", new_rels + "</Relationships>"), encoding="utf8")

    cpath = work / "[Content_Types].xml"
    ct = cpath.read_text(encoding="utf8")
    WML = "application/vnd.openxmlformats-officedocument.wordprocessingml"
    ct = ct.replace("</Types>",
                    '<Default Extension="png" ContentType="image/png"/>'
                    f'<Override PartName="/word/footer1.xml" ContentType="{WML}.footer+xml"/>'
                    f'<Override PartName="/word/numbering.xml" ContentType="{WML}.numbering+xml"/></Types>')
    cpath.write_text(ct, encoding="utf8")

    # --- zip: [Content_Types].xml first ---
    out_path = Path(out_path)
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(cpath, "[Content_Types].xml")
        for f in sorted(work.rglob("*")):
            if f.is_file() and f != cpath:
                z.write(f, f.relative_to(work).as_posix())
    shutil.rmtree(work)
    return headings


def heading_pages(docx_path, headings):
    """Render to PDF and find the printed page number of each heading."""
    outdir = Path(docx_path).parent
    subprocess.run([sys.executable, SOFFICE, "--headless", "--convert-to", "pdf", "--outdir", str(outdir),
                    str(docx_path)], check=True, capture_output=True)
    pdf = Path(docx_path).with_suffix(".pdf")
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True,
                          text=True).stdout
    pages = text.split("\f")
    found = {}
    start = None
    for idx, page in enumerate(pages):
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if not lines or "Contents" in lines[0]:
            continue
        for h in headings:
            if h not in found and lines and lines[0].startswith(h):
                found[h] = idx + 1
                if start is None:
                    start = idx + 1
    # Body numbering starts at 1 on the first heading's page
    return {h: p - start + 1 for h, p in found.items()}, len(pages) - 1


if __name__ == "__main__":
    template, figures, out = sys.argv[1:4]
    heads = build(template, figures, out)
    pages, _ = heading_pages(out, heads)
    build(template, figures, out, pages)
    pages2, n_pages = heading_pages(out, heads)
    assert pages == pages2, (pages, pages2)
    missing = [h for h in heads if h not in pages]
    print(json.dumps(pages, indent=1))
    print("pages in PDF:", n_pages, "| headings not found:", missing or "none")
