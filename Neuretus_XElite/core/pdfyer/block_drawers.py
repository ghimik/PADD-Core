import os
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

from .text_utils import fit_text_to_bbox
from .table_parser import parse_table_from_html, normalize_table_data


class BlockDrawer:
    """Базовый класс для рисовальщиков блоков"""
    
    def __init__(self, font_name: str, image_dir: str = None):
        self.font_name = font_name
        self.image_dir = image_dir
    
    def draw(self, c: canvas.Canvas, block: dict, page_h: float):
        """
        Рисует блок на PDF канве.
        
        Args:
            c: ReportLab canvas
            block: словарь с данными блока
            page_h: высота страницы (для преобразования координат)
        """
        raise NotImplementedError


class TextBlockDrawer(BlockDrawer):
    """Рисовальщик текстовых блоков"""
    
    def draw(self, c: canvas.Canvas, block: dict, page_h: float):
        x1, y1, x2, y2 = block["block_bbox"]
        w = x2 - x1
        h = y2 - y1

        text = block.get("block_content", "").strip()
        if not text:
            return

        fitted = fit_text_to_bbox(text, self.font_name, w, h)
        if fitted is None:
            return

        font_size, lines = fitted
        line_h = font_size * 1.2

        c.setFont(self.font_name, font_size)

        y = page_h - y1 - font_size
        for line in lines:
            c.drawString(x1, y, line)
            y -= line_h


class ImageBlockDrawer(BlockDrawer):
    """Рисовальщик блоков с изображениями"""
    
    def draw(self, c: canvas.Canvas, block: dict, page_h: float):
        x1, y1, x2, y2 = block["block_bbox"]
        w = x2 - x1
        h = y2 - y1

        label = block.get("block_label", "")

        if not self.image_dir:
            print(f"No image directory set, skipping image block")
            return

        img_name = f"img_in_{label}_box_{x1}_{y1}_{x2}_{y2}.jpg"
        img_path = os.path.join(self.image_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            return

        c.drawImage(
            img_path,
            x1,
            page_h - y2,
            width=w,
            height=h,
            preserveAspectRatio=False,
            mask="auto",
        )


class TableBlockDrawer(BlockDrawer):
    """Рисовальщик табличных блоков"""
    
    def __init__(self, font_name: str, max_rows: int = 50, image_dir: str = None):
        super().__init__(font_name, image_dir)
        self.max_rows = max_rows
    
    def draw(self, c: canvas.Canvas, block: dict, page_h: float):
        x1, y1, x2, y2 = block["block_bbox"]
        w = x2 - x1
        h = y2 - y1

        
        html_content = block.get("block_content", "")
        table_data = parse_table_from_html(html_content)
        
        if not table_data:
            
            print(f"Could not parse table, drawing as text instead")
            TextBlockDrawer(self.font_name).draw(c, block, page_h)
            return

        
        table_data = normalize_table_data(table_data)
        
        
        if len(table_data) > self.max_rows:
            print(f"Table too large ({len(table_data)} rows), truncating to {self.max_rows} rows")
            table_data = table_data[:self.max_rows]
            
            h = h * self.max_rows / len(table_data)
        
        try:
            num_rows = len(table_data)
            num_cols = len(table_data[0]) if num_rows > 0 else 1
            
            
            table = Table(table_data)
            
            
            col_widths = [w / num_cols] * num_cols
            row_heights = [h / num_rows] * num_rows
            
            table._argW = col_widths
            table._argH = row_heights
            
            
            font_size = min(12, h / (num_rows * 2))  
            table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, -1), self.font_name),
                ('FONTSIZE', (0, 0), (-1, -1), font_size),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
                ('TOPPADDING', (0, 0), (-1, -1), 1),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ]))
            
            
            table.drawOn(c, x1, page_h - y2)
            
        except Exception as e:
            print(f"Error drawing table: {e}")
            
            TextBlockDrawer(self.font_name).draw(c, block, page_h)


def get_block_drawer(block_type: str, font_name: str, image_dir: str = None) -> BlockDrawer:
    """
    Фабрика для создания рисовальщика по типу блока.
    """
    if "image" in block_type:
        return ImageBlockDrawer(font_name, image_dir)
    elif "table" in block_type:
        return TableBlockDrawer(font_name, image_dir=image_dir)
    else:
        return TextBlockDrawer(font_name, image_dir)