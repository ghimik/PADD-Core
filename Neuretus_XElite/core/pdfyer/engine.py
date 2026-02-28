import json
import os
import logging
from typing import Union, Dict, Any, Optional, List
from pathlib import Path

from bs4 import BeautifulSoup
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4


LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class PDFEngine:
    """
    PDF реконструктор (A4)
    """

    TEXT_LABELS = {
        "text",
        "paragraph_title",
        "header",
        "doc_title",
        "algorithm",
        "number",
        "footer",
        "aside_text",
        "table", 
    }

    IMAGE_LABELS = {
        "image",
        "header_image",
        "footer_image",
        "figure",
    }


    def __init__(self, font_path: str, font_name: str = "CustomFont"):
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found: {font_path}")

        self.font_path = font_path
        self.font_name = font_name
        pdfmetrics.registerFont(TTFont(font_name, font_path))


    def reconstruct(
        self,
        json_path: Union[str, Path],
        output_pdf: Union[str, Path],
        image_dir: Optional[str] = None,
    ) -> None:
        json_path = Path(json_path)
        output_pdf = Path(output_pdf)

        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        src_w = float(doc.get("width", 2481))
        src_h = float(doc.get("height", 3507))

        page_w, page_h = A4

        scale_x = page_w / src_w
        scale_y = page_h / src_h

        c = canvas.Canvas(str(output_pdf), pagesize=A4)

        for block in doc.get("parsing_res_list", []):
            try:
                self._render_block(
                    canvas_obj=c,
                    block=block,
                    page_w=page_w,
                    page_h=page_h,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    image_dir=image_dir,
                )
            except Exception:
                LOG.exception("Failed to render block: %s", block)

        c.showPage()
        c.save()


    def _render_block(
        self,
        canvas_obj: canvas.Canvas,
        block: Dict[str, Any],
        page_w: float,
        page_h: float,
        scale_x: float,
        scale_y: float,
        image_dir: Optional[str],
    ) -> None:
        label = block.get("block_label", "").lower()

        if label == "table":
            self._render_table_block(
                canvas_obj, block, page_h, scale_x, scale_y
            )
        elif label in self.TEXT_LABELS:
            self._render_text_block(
                canvas_obj, block, page_h, scale_x, scale_y
            )
        elif label in self.IMAGE_LABELS:
            self._render_image_block(
                canvas_obj, block, page_h, scale_x, scale_y, image_dir
            )
        else:
            return


    def _render_text_block(
        self,
        canvas_obj: canvas.Canvas,
        block: Dict[str, Any],
        page_h: float,
        scale_x: float,
        scale_y: float,
    ) -> None:
        content = block.get("block_content", "")
        if not content:
            return

        bbox = block.get("block_bbox")
        if not bbox or len(bbox) != 4:
            return

        x0, y0, x1, y1 = bbox

        x0 *= scale_x
        x1 *= scale_x
        y0 *= scale_y
        y1 *= scale_y

        width = x1 - x0
        height = y1 - y0

        font_size, lines = self._fit_text_into_box(
            text=content,
            box_width=width,
            box_height=height,
        )

        leading = font_size * 1.15

        top_pdf = page_h - y0
        bottom_pdf = page_h - y1

        first_baseline = top_pdf - font_size * 0.2

        canvas_obj.setFont(self.font_name, font_size)

        for i, line in enumerate(lines):
            y = first_baseline - i * leading
            if y < bottom_pdf:
                break
            canvas_obj.drawString(x0 + 1.0, y, line)


    def _render_image_block(
        self,
        canvas_obj: canvas.Canvas,
        block: Dict[str, Any],
        page_h: float,
        scale_x: float,
        scale_y: float,
        image_dir: Optional[str],
    ) -> None:
        if not image_dir:
            return

        bbox = block.get("block_bbox")
        if not bbox or len(bbox) != 4:
            return

        x0_raw, y0_raw, x1_raw, y1_raw = bbox

        label = block.get("block_label", "").lower()


        filename = f"img_in_{label}_box_{int(x0_raw)}_{int(y0_raw)}_{int(x1_raw)}_{int(y1_raw)}.jpg"

        image_path = Path(image_dir) / filename
        if not image_path.exists():
            LOG.warning("Image not found: %s", image_path)
            return


        bbox = block.get("block_bbox")
        if not bbox or len(bbox) != 4:
            return

        x0, y0, x1, y1 = bbox

        x0 *= scale_x
        x1 *= scale_x
        y0 *= scale_y
        y1 *= scale_y

        width = x1 - x0
        height = y1 - y0

        pdf_y = page_h - y1

        canvas_obj.drawImage(
            str(image_path),
            x0,
            pdf_y,
            width=width,
            height=height,
            preserveAspectRatio=False,
            mask="auto",
        )


    def _fit_text_into_box(
        self,
        text: str,
        box_width: float,
        box_height: float,
        max_font: float = 24.0,
        min_font: float = 6.0,
    ) -> tuple[float, List[str]]:
        paragraphs = text.split("\n")

        font_size = max_font
        while font_size >= min_font:
            lines: List[str] = []
            for p in paragraphs:
                lines.extend(
                    self._wrap_text(p.strip(), font_size, box_width - 2)
                )

            required_height = len(lines) * font_size * 1.15
            if required_height <= box_height:
                return font_size, lines

            font_size *= 0.9

        lines = []
        for p in paragraphs:
            lines.extend(self._wrap_text(p.strip(), min_font, box_width - 2))
        return min_font, lines

    def _wrap_text(
        self,
        text: str,
        font_size: float,
        max_width: float,
    ) -> List[str]:
        if not text:
            return [""]

        words = text.split()
        lines: List[str] = []
        current: List[str] = []

        for word in words:
            candidate = " ".join(current + [word])
            width = pdfmetrics.stringWidth(
                candidate, self.font_name, font_size
            )

            if width <= max_width or not current:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]

        if current:
            lines.append(" ".join(current))

        return lines
    

    def _render_table_block(
        self,
        canvas_obj: canvas.Canvas,
        block: Dict[str, Any],
        page_h: float,
        scale_x: float,
        scale_y: float,
    ) -> None:
        html = block.get("block_content", "")
        if not html:
            return

        bbox = block.get("block_bbox")
        if not bbox or len(bbox) != 4:
            return

        x0_raw, y0_raw, x1_raw, y1_raw = bbox

        x0 = x0_raw * scale_x
        x1 = x1_raw * scale_x
        y0 = y0_raw * scale_y
        y1 = y1_raw * scale_y

        table_width = x1 - x0
        table_height = y1 - y0

        cells, n_rows, n_cols = self._parse_uniform_table(html)

        if n_rows == 0 or n_cols == 0:
            return

        col_width = table_width / n_cols
        row_height = table_height / n_rows

        for cell in cells:
            r = cell["row"]
            c = cell["col"]
            colspan = cell["colspan"]
            text = cell["text"]

            cell_x0 = x0 + c * col_width
            cell_x1 = x0 + (c + colspan) * col_width

            cell_y0 = y0 + r * row_height
            cell_y1 = cell_y0 + row_height

            pdf_y = page_h - cell_y1

            # canvas_obj.rect(
            #     cell_x0,
            #     pdf_y,
            #     cell_x1 - cell_x0,
            #     row_height,
            #     stroke=1,
            #     fill=0,
            # )

            if text:
                font_size, lines = self._fit_text_into_box(
                    text,
                    cell_x1 - cell_x0,
                    row_height,
                    max_font=12,
                )

                canvas_obj.setFont(self.font_name, font_size)

                leading = font_size * 1.15
                top_pdf = page_h - cell_y0
                first_baseline = top_pdf - font_size * 0.2

                for i, line in enumerate(lines):
                    y = first_baseline - i * leading
                    if y < pdf_y:
                        break
                    canvas_obj.drawString(cell_x0 + 2, y, line)
    

    def _parse_uniform_table(self, html: str):

        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")

        n_rows = len(rows)
        if n_rows == 0:
            return [], 0, 0

        n_cols = 0
        for tr in rows:
            count = 0
            for td in tr.find_all("td"):
                colspan = int(td.get("colspan", 1))
                count += colspan
            n_cols = max(n_cols, count)

        cells = []

        for r, tr in enumerate(rows):
            c = 0
            for td in tr.find_all("td"):
                colspan = int(td.get("colspan", 1))
                text = td.get_text(strip=True)

                cells.append({
                    "row": r,
                    "col": c,
                    "colspan": colspan,
                    "text": text,
                })

                c += colspan

        return cells, n_rows, n_cols
