import re
from html.parser import HTMLParser


class TableHTMLParser(HTMLParser):
    """Парсер HTML таблицы с поддержкой rowspan и colspan"""
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current_row = []
        self.current_cell = ""
        self.in_cell = False
        
    def handle_starttag(self, tag, attrs):
        if tag in ['td', 'th']:
            self.in_cell = True
            self.current_cell = ""
            
    def handle_endtag(self, tag):
        if tag in ['td', 'th']:
            self.in_cell = False
            
            cell_text = self.current_cell.strip()
            cell_text = re.sub(r'\s+', ' ', cell_text)
            self.current_row.append(cell_text)
        elif tag == 'tr':
            if self.current_row:
                self.rows.append(self.current_row)
                self.current_row = []
                
    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data


def parse_html_table_simple(html_content: str) -> list:
    """Простой парсинг HTML таблицы - извлекаем только текст построчно"""
    rows_text = []
    
    tr_matches = re.findall(r'<tr>(.*?)</tr>', html_content, re.DOTALL)
    
    for tr_match in tr_matches:
        row_cells = []
        
        td_matches = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', tr_match, re.DOTALL)
        
        for td_content in td_matches:
            
            cell_text = re.sub(r'<[^>]+>', ' ', td_content)
            cell_text = re.sub(r'\s+', ' ', cell_text).strip()
            row_cells.append(cell_text)
        
        if row_cells:
            rows_text.append(row_cells)
    
    return rows_text


def parse_table_from_html(html_content: str) -> list:
    """
    Основная функция парсинга таблицы.
    
    Returns:
        список строк, каждая строка - список ячеек
    """
    if not html_content or not html_content.strip():
        return []
    
    
    parser = TableHTMLParser()
    parser.feed(html_content)
    
    if parser.rows:
        return parser.rows
    
    
    return parse_html_table_simple(html_content)


def normalize_table_data(table_data: list, max_cols: int = None) -> list:
    """
    Нормализует таблицу: все строки одинаковой длины, пустые ячейки заполнены.
    
    Args:
        table_data: сырые данные таблицы
        max_cols: максимальное количество столбцов (если None, вычисляется)
        
    Returns:
        нормализованная таблица
    """
    if not table_data:
        return []
    
    if max_cols is None:
        max_cols = max(len(row) for row in table_data) if table_data else 1
    
    normalized = []
    for row in table_data:
        if len(row) < max_cols:
            row = row + [''] * (max_cols - len(row))
        normalized.append(row)
    
    return normalized