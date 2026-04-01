import json
import logging
import os
import re
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class PDFEngine:
    """
    PDF реконструктор (A4)
    """

    LINE_HEIGHT = 1.15
    DEFAULT_PADDING_X = 2.0
    DEFAULT_PADDING_Y = 1.5

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

    LATEX_COMMAND_REPLACEMENTS = {
        r"\pm": "±",
        r"\mp": "∓",
        r"\times": "×",
        r"\cdot": "·",
        r"\div": "÷",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\sim": "~",
        r"\to": "→",
        r"\rightarrow": "→",
        r"\leftarrow": "←",
        r"\leftrightarrow": "↔",
        r"\degree": "°",
        r"\circ": "°",
        r"\infty": "∞",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\epsilon": "ε",
        r"\theta": "θ",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\pi": "π",
        r"\sigma": "σ",
        r"\phi": "φ",
        r"\omega": "ω",
        r"\Omega": "Ω",
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
                canvas_obj, block, page_w, page_h, scale_x, scale_y
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
        page_w: float,
        page_h: float,
        scale_x: float,
        scale_y: float,
    ) -> None:
        content = self._normalize_text_content(
            block.get("block_content", "")
        )
        if not content:
            return

        scaled_bbox = self._scale_bbox(
            block.get("block_bbox"),
            scale_x,
            scale_y,
        )
        if not scaled_bbox:
            return

        x0, y0, x1, y1 = scaled_bbox
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            return

        bottom_pdf = page_h - y1
        is_vertical = self._should_render_vertical_text(
            block=block,
            box_width=width,
            box_height=height,
            text=content,
        )
        rotate_clockwise = (x0 + width / 2.0) >= (page_w / 2.0)

        self._render_text_box(
            canvas_obj=canvas_obj,
            text=content,
            x=x0,
            y=bottom_pdf,
            box_width=width,
            box_height=height,
            max_font=24.0,
            min_font=6.0,
            is_vertical=is_vertical,
            rotate_clockwise=rotate_clockwise,
        )

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

        scaled_bbox = self._scale_bbox(
            bbox,
            scale_x,
            scale_y,
        )
        if not scaled_bbox:
            return

        x0, y0, x1, y1 = scaled_bbox
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
        min_font: float = 4.0,
    ) -> Tuple[float, List[str]]:
        if box_width <= 0 or box_height <= 0:
            return min_font, []

        font_size = max(max_font, min_font)
        while font_size >= (min_font - 0.01):
            lines = self._wrap_paragraphs(text, font_size, box_width)
            required_height = self._required_text_height(lines, font_size)
            if required_height <= box_height + 0.01:
                return font_size, lines

            next_font = font_size * 0.9
            if next_font < min_font:
                break
            font_size = next_font

        lines = self._wrap_paragraphs(text, min_font, box_width)
        fitted_lines = self._truncate_lines_to_height(
            lines=lines,
            font_size=min_font,
            box_height=box_height,
            box_width=box_width,
        )
        return min_font, fitted_lines

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
        current = ""

        for word in words:
            word_chunks = self._split_long_token(
                token=word,
                font_size=font_size,
                max_width=max_width,
            )

            if len(word_chunks) > 1:
                if current:
                    lines.append(current)
                    current = ""
                lines.extend(word_chunks[:-1])
                current = word_chunks[-1]
                continue

            chunk = word_chunks[0]
            candidate = chunk if not current else f"{current} {chunk}"
            if self._measure_text(candidate, font_size) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = chunk

        if current:
            lines.append(current)

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

        scaled_bbox = self._scale_bbox(
            block.get("block_bbox"),
            scale_x,
            scale_y,
        )
        if not scaled_bbox:
            return

        x0, y0, x1, y1 = scaled_bbox
        table_width = x1 - x0
        table_height = y1 - y0
        if table_width <= 0 or table_height <= 0:
            return

        cells, n_rows, n_cols = self._parse_uniform_table(html)

        if n_rows == 0 or n_cols == 0:
            return

        font_size, col_widths, row_heights = self._fit_table_layout(
            cells=cells,
            n_rows=n_rows,
            n_cols=n_cols,
            table_width=table_width,
            table_height=table_height,
            max_font=12.0,
            min_font=7.0,
        )

        col_offsets = [x0]
        for col_width in col_widths:
            col_offsets.append(col_offsets[-1] + col_width)

        row_offsets = [y0]
        for row_height in row_heights:
            row_offsets.append(row_offsets[-1] + row_height)

        for cell in cells:
            r = cell["row"]
            c = cell["col"]
            colspan = cell["colspan"]
            rowspan = cell["rowspan"]
            text = cell["text"]

            cell_x0 = col_offsets[c]
            cell_x1 = col_offsets[min(c + colspan, len(col_offsets) - 1)]
            cell_y0 = row_offsets[r]
            cell_y1 = row_offsets[min(r + rowspan, len(row_offsets) - 1)]

            if not text:
                continue

            self._render_text_box(
                canvas_obj=canvas_obj,
                text=text,
                x=cell_x0,
                y=page_h - cell_y1,
                box_width=cell_x1 - cell_x0,
                box_height=cell_y1 - cell_y0,
                max_font=font_size,
                min_font=font_size,
            )

    def _parse_uniform_table(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            return [], 0, 0

        cells = []
        occupied: List[Dict[int, bool]] = []
        max_cols = 0

        for r, tr in enumerate(rows):
            while len(occupied) <= r:
                occupied.append({})

            c = 0
            for td in tr.find_all(["td", "th"]):
                while occupied[r].get(c):
                    c += 1

                colspan = self._safe_int(td.get("colspan"), 1)
                rowspan = self._safe_int(td.get("rowspan"), 1)
                text = self._normalize_text_content(
                    td.get_text("\n", strip=True)
                )

                cells.append(
                    {
                        "row": r,
                        "col": c,
                        "colspan": colspan,
                        "rowspan": rowspan,
                        "text": text,
                    }
                )

                for rr in range(r, r + rowspan):
                    while len(occupied) <= rr:
                        occupied.append({})
                    for cc in range(c, c + colspan):
                        occupied[rr][cc] = True

                c += colspan
                max_cols = max(max_cols, c)

        return cells, len(rows), max_cols

    def _render_text_box(
        self,
        canvas_obj: canvas.Canvas,
        text: str,
        x: float,
        y: float,
        box_width: float,
        box_height: float,
        max_font: float,
        min_font: float,
        is_vertical: bool = False,
        rotate_clockwise: bool = False,
    ) -> None:
        if not text or box_width <= 0 or box_height <= 0:
            return

        padding_x = min(self.DEFAULT_PADDING_X, max(0.5, box_width * 0.02))
        padding_y = min(self.DEFAULT_PADDING_Y, max(0.5, box_height * 0.02))

        local_width = box_height if is_vertical else box_width
        local_height = box_width if is_vertical else box_height

        content_width = max(local_width - 2 * padding_x, 1.0)
        content_height = max(local_height - 2 * padding_y, 1.0)

        font_size, lines = self._fit_text_into_box(
            text=text,
            box_width=content_width,
            box_height=content_height,
            max_font=max_font,
            min_font=min_font,
        )
        if not lines:
            return

        canvas_obj.saveState()
        self._clip_rect(canvas_obj, x, y, box_width, box_height)

        if is_vertical:
            if rotate_clockwise:
                canvas_obj.translate(x, y + box_height)
                canvas_obj.rotate(-90)
            else:
                canvas_obj.translate(x + box_width, y)
                canvas_obj.rotate(90)
        else:
            canvas_obj.translate(x, y)

        self._draw_text_lines_local(
            canvas_obj=canvas_obj,
            lines=lines,
            font_size=font_size,
            box_width=local_width,
            box_height=local_height,
            padding_x=padding_x,
            padding_y=padding_y,
        )
        canvas_obj.restoreState()

    def _draw_text_lines_local(
        self,
        canvas_obj: canvas.Canvas,
        lines: List[str],
        font_size: float,
        box_width: float,
        box_height: float,
        padding_x: float,
        padding_y: float,
    ) -> None:
        if not lines:
            return

        canvas_obj.setFont(self.font_name, font_size)

        leading = font_size * self.LINE_HEIGHT
        first_baseline = box_height - padding_y - font_size * 0.85
        min_baseline = padding_y + font_size * 0.2

        for index, line in enumerate(lines):
            y = first_baseline - index * leading
            if y < min_baseline:
                break
            canvas_obj.drawString(padding_x, y, line)

    def _clip_rect(
        self,
        canvas_obj: canvas.Canvas,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        path = canvas_obj.beginPath()
        path.rect(x, y, width, height)
        canvas_obj.clipPath(path, stroke=0, fill=0)

    def _required_text_height(
        self,
        lines: List[str],
        font_size: float,
    ) -> float:
        if not lines:
            return 0.0
        return len(lines) * font_size * self.LINE_HEIGHT

    def _truncate_lines_to_height(
        self,
        lines: List[str],
        font_size: float,
        box_height: float,
        box_width: float,
    ) -> List[str]:
        if not lines:
            return []

        max_lines = max(
            int(box_height // (font_size * self.LINE_HEIGHT)),
            1,
        )
        if len(lines) <= max_lines:
            return lines

        fitted = lines[:max_lines]
        last_line = fitted[-1].rstrip()
        ellipsis = "..."
        while last_line and (
            self._measure_text(last_line + ellipsis, font_size) > box_width
        ):
            last_line = last_line[:-1].rstrip()
        fitted[-1] = (last_line + ellipsis) if last_line else ellipsis
        return fitted

    def _wrap_paragraphs(
        self,
        text: str,
        font_size: float,
        max_width: float,
    ) -> List[str]:
        if max_width <= 0:
            return []

        lines: List[str] = []
        paragraphs = text.splitlines() or [text]

        for paragraph in paragraphs:
            stripped = paragraph.strip()
            if not stripped:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            lines.extend(self._wrap_text(stripped, font_size, max_width))

        return lines or [""]

    def _split_long_token(
        self,
        token: str,
        font_size: float,
        max_width: float,
    ) -> List[str]:
        if not token:
            return [""]
        if self._measure_text(token, font_size) <= max_width:
            return [token]

        chunks: List[str] = []
        current = ""

        for char in token:
            candidate = current + char
            if current and self._measure_text(candidate, font_size) > max_width:
                chunks.append(current)
                current = char
            else:
                current = candidate

        if current:
            chunks.append(current)

        return chunks or [token]

    def _measure_text(self, text: str, font_size: float) -> float:
        return pdfmetrics.stringWidth(text, self.font_name, font_size)

    def _fit_table_layout(
        self,
        cells: List[Dict[str, Any]],
        n_rows: int,
        n_cols: int,
        table_width: float,
        table_height: float,
        max_font: float,
        min_font: float,
    ) -> Tuple[float, List[float], List[float]]:
        col_widths = [table_width / n_cols for _ in range(n_cols)]
        best_row_heights = [table_height / n_rows for _ in range(n_rows)]

        font_size = max(max_font, min_font)
        while font_size >= (min_font - 0.01):
            row_heights = self._estimate_row_heights(
                cells=cells,
                n_rows=n_rows,
                col_widths=col_widths,
                font_size=font_size,
            )
            total_height = sum(row_heights)
            if total_height <= table_height + 0.01:
                return font_size, col_widths, row_heights

            best_row_heights = row_heights
            next_font = font_size * 0.92
            if next_font < min_font:
                break
            font_size = next_font

        scaled_row_heights = self._scale_row_heights_to_fit(
            row_heights=best_row_heights,
            target_height=table_height,
        )
        return min_font, col_widths, scaled_row_heights

    def _estimate_row_heights(
        self,
        cells: List[Dict[str, Any]],
        n_rows: int,
        col_widths: List[float],
        font_size: float,
    ) -> List[float]:
        base_height = font_size * self.LINE_HEIGHT + 2 * self.DEFAULT_PADDING_Y
        row_heights = [base_height for _ in range(n_rows)]

        for _ in range(max(n_rows * 3, 1)):
            changed = False
            for cell in cells:
                text = cell["text"]
                if not text:
                    continue

                row = cell["row"]
                rowspan = max(cell["rowspan"], 1)
                start = row
                end = min(row + rowspan, n_rows)

                cell_width = sum(
                    col_widths[cell["col"] : cell["col"] + cell["colspan"]]
                )
                available_width = max(
                    cell_width - 2 * self.DEFAULT_PADDING_X,
                    1.0,
                )
                lines = self._wrap_paragraphs(text, font_size, available_width)
                required_height = (
                    self._required_text_height(lines, font_size)
                    + 2 * self.DEFAULT_PADDING_Y
                )

                current_height = sum(row_heights[start:end])
                if current_height + 0.01 < required_height:
                    deficit = required_height - current_height
                    addition = deficit / max(end - start, 1)
                    for row_index in range(start, end):
                        row_heights[row_index] += addition
                    changed = True

            if not changed:
                break

        return row_heights

    def _scale_row_heights_to_fit(
        self,
        row_heights: List[float],
        target_height: float,
    ) -> List[float]:
        if not row_heights:
            return []

        current_height = sum(row_heights)
        if current_height <= 0:
            return row_heights

        scale = target_height / current_height
        scaled = [height * scale for height in row_heights]
        scaled[-1] += target_height - sum(scaled)
        return scaled

    def _should_render_vertical_text(
        self,
        block: Dict[str, Any],
        box_width: float,
        box_height: float,
        text: str,
    ) -> bool:
        compact_text = re.sub(r"\s+", "", text)
        if not compact_text:
            return False

        label = block.get("block_label", "").lower()
        if label == "aside_text" and box_height > box_width * 1.2:
            return True

        if "\n" in text:
            return False
        if box_height <= box_width * 1.35:
            return False
        if len(compact_text) > 24:
            return False
        if len(text.split()) > 3:
            return False
        return True

    def _normalize_text_content(self, text: Any) -> str:
        if text is None:
            return ""

        normalized = unescape(str(text))
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\xa0", " ")

        if "$" in normalized or "\\" in normalized:
            normalized = self._normalize_latex(normalized)
        normalized = self._collapse_suspicious_runs(normalized)

        normalized_lines: List[str] = []
        for line in normalized.split("\n"):
            cleaned = re.sub(r"[ \t\f\v]+", " ", line).strip()
            if not cleaned:
                if normalized_lines and normalized_lines[-1] != "":
                    normalized_lines.append("")
                continue
            normalized_lines.append(cleaned)

        return "\n".join(normalized_lines).strip()

    def _normalize_latex(self, text: str) -> str:
        normalized = text.replace("$$", "$")
        normalized = normalized.replace(r"\(", "$")
        normalized = normalized.replace(r"\)", "$")
        normalized = normalized.replace(r"\[", "$")
        normalized = normalized.replace(r"\]", "$")
        normalized = re.sub(
            r"\$([^$]+)\$",
            lambda match: self._normalize_latex_fragment(match.group(1)),
            normalized,
        )
        return self._normalize_latex_fragment(normalized)

    def _normalize_latex_fragment(self, text: str) -> str:
        result = text.replace("\\\\", "\n")
        wrapper_names = (
            "mathrm|text|operatorname|mathbf|mathit|mathsf|mbox|rm"
        )

        previous = None
        while previous != result:
            previous = result
            result = re.sub(
                r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
                r"(\1)/(\2)",
                result,
            )
            result = re.sub(
                r"\\sqrt\s*\{([^{}]+)\}",
                r"sqrt(\1)",
                result,
            )
            result = re.sub(
                rf"\\(?:{wrapper_names})\s*\{{([^{{}}]+)\}}",
                r"\1",
                result,
            )
            result = re.sub(r"_\{([^{}]+)\}", r"_\1", result)
            result = re.sub(r"\^\{([^{}]+)\}", r"^\1", result)

        for command, replacement in self.LATEX_COMMAND_REPLACEMENTS.items():
            result = result.replace(command, replacement)

        result = re.sub(r"\\([A-Za-z]+)", lambda match: match.group(1), result)
        result = re.sub(r"\\([{}_$%&#])", r"\1", result)
        result = result.replace("{", "").replace("}", "")
        result = result.replace("~", " ")
        result = result.replace("&", " ")
        result = re.sub(r"[ \t]+", " ", result)
        result = re.sub(r" *\n *", "\n", result)
        return result

    def _scale_bbox(
        self,
        bbox: Any,
        scale_x: float,
        scale_y: float,
    ) -> Optional[Tuple[float, float, float, float]]:
        if not bbox or len(bbox) != 4:
            return None

        x0, y0, x1, y1 = bbox
        return (
            x0 * scale_x,
            y0 * scale_y,
            x1 * scale_x,
            y1 * scale_y,
        )

    def _collapse_suspicious_runs(self, text: str) -> str:
        return re.sub(
            r"([0-9A-Za-zА-Яа-я])\1{31,}",
            lambda match: match.group(1) * 24 + "...",
            text,
        )

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return max(int(value), 1)
        except (TypeError, ValueError):
            return default
