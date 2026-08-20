"""
Populate a PPTX template from a CSV file.

Each CSV column header is matched to a shape by:
  1. Shape name (as shown in the Selection Pane)
  2. Shape text content (e.g. {{Presentation Title}})

Usage:
  python populate_pptx.py template.pptx data.csv output.pptx
"""

import csv
import re
import shutil
import sys
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.oxml.ns import qn
import openpyxl

from A_Config import case_dir, report_path, asset_list_path, assets_dir, asset_name


def set_text(shape, value):
    tf = shape.text_frame
    # Find the first run to preserve its formatting
    first_run = None
    for para in tf.paragraphs:
        for run in para.runs:
            if first_run is None:
                first_run = run
            else:
                run.text = ""
    if first_run is not None:
        first_run.text = value
    else:
        tf.text = value


PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def substitute_placeholders(shape, report):
    '''Replace every {{key}} token in shape's text with report[key] (left
    as-is if key isn't in the report). Lets one text box hold several
    independent values regardless of the shape's own name -- returns True if
    the shape had any tokens (and so has already been fully handled).'''
    text = shape.text_frame.text
    if not PLACEHOLDER_RE.search(text):
        return False
    new_text = PLACEHOLDER_RE.sub(lambda m: report.get(m.group(0), m.group(0)), text)
    set_text(shape, new_text)
    return True


def replace_image(slide, shape, image_path):
    blip = shape._element.find(".//" + qn("a:blip"))
    if blip is None:
        print(f"  Warning: shape '{shape.name}' has no image to replace")
        return
    new_img_part, new_rId = slide.part.get_or_add_image_part(image_path)
    blip.set(qn("r:embed"), new_rId)


def replace_video(slide, shape, video_path):
    video_file_el = shape._element.find(".//" + qn("a:videoFile"))
    if video_file_el is None:
        print(f"  Warning: no videoFile element in shape '{shape.name}'")
        return
    rId = video_file_el.get(qn("r:link"))
    media_part = slide.part.related_part(rId)
    with open(video_path, "rb") as f:
        media_part._blob = f.read()


def _col_letter(n):
    result = ""
    while n:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _update_range_end_row(ref, end_row):
    return re.sub(r"(\$[A-Z]+\$)\d+$", lambda m: m.group(1) + str(end_row), ref)


def _populate_num_cache(cache_el, ns, values):
    from lxml import etree
    for child in list(cache_el):
        cache_el.remove(child)
    fmt = etree.SubElement(cache_el, f"{{{ns}}}formatCode")
    fmt.text = "General"
    pt_count = etree.SubElement(cache_el, f"{{{ns}}}ptCount")
    pt_count.set("val", str(len(values)))
    for i, v in enumerate(values):
        pt = etree.SubElement(cache_el, f"{{{ns}}}pt")
        pt.set("idx", str(i))
        val_el = etree.SubElement(pt, f"{{{ns}}}v")
        val_el.text = str(v)


def _populate_str_cache(cache_el, ns, value):
    from lxml import etree
    for child in list(cache_el):
        cache_el.remove(child)
    pt_count = etree.SubElement(cache_el, f"{{{ns}}}ptCount")
    pt_count.set("val", "1")
    pt = etree.SubElement(cache_el, f"{{{ns}}}pt")
    pt.set("idx", "0")
    val_el = etree.SubElement(pt, f"{{{ns}}}v")
    val_el.text = str(value)


def update_scatter(shape, csv_path):
    ns = "http://schemas.openxmlformats.org/drawingml/2006/chart"

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        print(f"  Warning: chart CSV '{csv_path}' is empty")
        return

    header = rows[0]
    n_data = len(rows) - 1
    end_row = n_data + 1
    n_series = len(header) - 1

    # Build a new workbook from the CSV and swap the blob
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row, start=1):
            if r_idx == 1:
                ws.cell(row=r_idx, column=c_idx).value = val
            else:
                try:
                    ws.cell(row=r_idx, column=c_idx).value = float(val)
                except (ValueError, TypeError):
                    ws.cell(row=r_idx, column=c_idx).value = val

    buf = BytesIO()
    wb.save(buf)
    chart = shape.chart
    chart.part.chart_workbook.xlsx_part._blob = buf.getvalue()

    # Update chart XML range references and caches
    scatter_chart = chart._element.find(f".//{{{ns}}}scatterChart")
    existing_series = scatter_chart.findall(f"{{{ns}}}ser")

    x_values = [float(r[0]) for r in rows[1:]]

    for ser_idx, ser in enumerate(existing_series):
        for f_el in ser.iter(f"{{{ns}}}f"):
            f_el.text = _update_range_end_row(f_el.text, end_row)
        y_values = [float(r[ser_idx + 1]) for r in rows[1:]]
        series_name = header[ser_idx + 1]
        for cache_el in ser.findall(f".//{{{ns}}}xVal//{{{ns}}}numCache"):
            _populate_num_cache(cache_el, ns, x_values)
        for cache_el in ser.findall(f".//{{{ns}}}yVal//{{{ns}}}numCache"):
            _populate_num_cache(cache_el, ns, y_values)
        for cache_el in ser.findall(f".//{{{ns}}}tx//{{{ns}}}strCache"):
            _populate_str_cache(cache_el, ns, series_name)

    # Add series for any extra Y columns beyond what the template has
    if n_series > len(existing_series):
        template_ser = existing_series[0]
        for i in range(len(existing_series), n_series):
            col = _col_letter(i + 2)
            new_ser = deepcopy(template_ser)
            new_ser.find(f"{{{ns}}}idx").set("val", str(i))
            new_ser.find(f"{{{ns}}}order").set("val", str(i))
            tx_f = new_ser.find(f".//{{{ns}}}tx//{{{ns}}}f")
            if tx_f is not None:
                tx_f.text = f"Sheet1!${col}$1"
            yval_f = new_ser.find(f".//{{{ns}}}yVal//{{{ns}}}f")
            if yval_f is not None:
                yval_f.text = f"Sheet1!${col}$2:${col}${end_row}"
            scatter_chart.append(new_ser)


def dispatch_chart(shape):
    ns = "http://schemas.openxmlformats.org/drawingml/2006/chart"
    chart_el = shape.chart._element
    if chart_el.find(f".//{{{ns}}}scatterChart") is not None:
        update_scatter(shape, report_path())
    else:
        print(f"  WARNING: chart type in shape '{shape.name}' is not yet supported")


def add_hyperlink(shape, target_path, output_dir):
    '''Make `shape` clickable, following the link to `target_path`. Copies
    the target next to the pptx in `output_dir` and links to it by relative
    filename, so the deliverable folder stays self-contained and portable.'''
    target_path = Path(target_path)
    dest_path = output_dir / target_path.name
    shutil.copy2(target_path, dest_path)
    shape.click_action.hyperlink.address = dest_path.name


def replace_video_thumb(slide, shape, image_path):
    blip = shape._element.find(".//" + qn("a:blip"))
    if blip is None:
        print(f"  Warning: no poster frame blip in shape '{shape.name}'")
        return
    new_img_part, new_rId = slide.part.get_or_add_image_part(image_path)
    blip.set(qn("r:embed"), new_rId)


def delete_slide(prs, slide):
    '''Remove `slide` from `prs`, including its slide-list entry and its
    relationship, so no orphaned part is left behind in the package.'''
    xml_slides = prs.slides._sldIdLst
    for sldId in list(xml_slides):
        rId = sldId.get(qn("r:id"))
        if prs.part.related_part(rId) is slide.part:
            prs.part.drop_rel(rId)
            xml_slides.remove(sldId)
            break


def _load_csv_dict(path):
    d = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            key, value = row[0].strip(), row[1].strip()
            if key and value:
                d[key] = value
    return d


PERSON_KEY_RE = re.compile(r"^person-(\d+)_(.+)$")


def _renumber_person_keys(report):
    '''person-N is Gemini's own person_id numbering -- stable per person
    within a run, and (per Gemini reading frames in order) already in
    first-appearance order, but not necessarily dense: a video's beats
    might only ever reference person-2 and person-5. Renumber to a dense
    1, 2, 3... run, preserving that same relative order, so a template can
    pre-author fixed "person-1_...", "person-2_..." slots regardless of
    which raw ids a given video happens to produce.'''
    raw_ids = sorted({int(m.group(1)) for k in report if (m := PERSON_KEY_RE.match(k))})
    id_map = {raw: i + 1 for i, raw in enumerate(raw_ids)}

    renumbered = {}
    for key, value in report.items():
        m = PERSON_KEY_RE.match(key)
        if m:
            key = f"person-{id_map[int(m.group(1))]}_{m.group(2)}"
        renumbered[key] = value
    return renumbered


def populate(report_template):

    output_path = case_dir() / "100_Powerpoint" / f"{asset_name()}.pptx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prs = Presentation(report_template)

    #every report key has a template object it COULD match, but any given
    #template only carries a subset of those objects -- so drive from the
    #objects actually present in this template and look up their value,
    #rather than looping every report row and asking "is there a shape for
    #this" (which warned on every key the template simply doesn't carry).
    #report + asset list are two separate CSVs upstream (data/decisions vs.
    #produced files) but the pptx template doesn't care which one a key came
    #from, so merge them back into one lookup here.
    report = _load_csv_dict(report_path())
    if asset_list_path().exists():
        report.update(_load_csv_dict(asset_list_path()))
    report = _renumber_person_keys(report)

    empty_slides = []
    for slide in prs.slides:
        slide_populated = False
        for shape in slide.shapes:
            if shape.has_text_frame and substitute_placeholders(shape, report):
                slide_populated = True
                continue

            #a "..._clicker" shape is matched on its own name as normal, but
            #also becomes a hyperlink to the "..._link" report key
            is_clicker = shape.name.endswith("_clicker")

            candidates = [shape.name]
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    candidates.append(text)

            base_key, is_thumb, value = None, False, None
            for cand in candidates:
                if cand in report:
                    base_key, is_thumb, value = cand, False, report[cand]
                    break
                if f"{cand}_thumb" in report:
                    base_key, is_thumb, value = cand, True, report[f"{cand}_thumb"]
                    break
            if value is None:
                continue  # this template doesn't carry an object for any report key

            slide_populated = True
            key = f"{base_key}_thumb" if is_thumb else base_key
            print(f"  '{key}' -> shape '{shape.name}'")

            if key == "Video":
                #bookkeeping only (source video's filename) -- always show as
                #text, never resolved as an assets_dir path / embedded media
                if shape.has_text_frame:
                    set_text(shape, value)
                else:
                    print(f"  WARNING: shape '{shape.name}' matched for 'Video' is a "
                          f"'{shape.__class__.__name__}', not a text shape -- skipping")
            elif is_thumb:
                replace_video_thumb(slide, shape, str(assets_dir() / value))
            elif shape.has_chart:
                dispatch_chart(shape, str(assets_dir() / value))
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                replace_image(slide, shape, str(assets_dir() / value))
            elif shape.__class__.__name__ == "Movie":
                replace_video(slide, shape, str(assets_dir() / value))
            elif shape.has_text_frame:
                if value.endswith(".txt"):
                    set_text(shape, (assets_dir() / value).read_text(encoding="utf-8"))
                else:
                    set_text(shape, value)
            else:
                print(f"  WARNING: shape '{shape.name}' is type '{shape.__class__.__name__}', skipping")

            #a video shape carries both a videoFile and a poster blip -- the
            #match above only ever resolves one value per shape, so apply the
            #poster separately whenever both keys exist for the same shape
            if shape.__class__.__name__ == "Movie" and f"{shape.name}_thumb" in report:
                replace_video_thumb(slide, shape, str(assets_dir() / report[f"{shape.name}_thumb"]))

            if is_clicker:
                link_key = f"{shape.name[:-len('_clicker')]}_link"
                if link_key in report:
                    add_hyperlink(shape, assets_dir() / report[link_key], output_path.parent)

        if not slide_populated:
            empty_slides.append(slide)

    for slide in empty_slides:
        print(f"  Removing empty slide (nothing matched from report)")
        delete_slide(prs, slide)

    prs.save(output_path)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    # CMD line use only: python populate_pptx.py template.pptx data.csv output.pptx assets_dir
    if len(sys.argv) != 5:
        print("Usage: python populate_pptx.py template.pptx data.csv output.pptx assets_dir")
        sys.exit(1)
    populate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
