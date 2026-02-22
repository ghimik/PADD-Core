from reportlab.pdfbase import pdfmetrics


def wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> list:
    """
    Разбивает текст на строки по ширине.
    
    Args:
        text: исходный текст
        font_name: имя шрифта
        font_size: размер шрифта
        max_width: максимальная ширина строки
        
    Returns:
        список строк
    """
    words = text.split()
    lines = []
    cur = ""

    for w in words:
        test = cur + (" " if cur else "") + w
        if pdfmetrics.stringWidth(test, font_name, font_size) <= max_width:
            cur = test
        else:
            if cur:  
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    return lines


def fit_text_to_bbox(text: str, font_name: str, bbox_w: float, bbox_h: float) -> tuple:
    """
    Подбирает максимальный размер шрифта, чтобы текст влез в bounding box.
    
    Args:
        text: текст
        font_name: имя шрифта
        bbox_w: ширина области
        bbox_h: высота области
        
    Returns:
        (font_size, lines) или None, если не влезает даже с размером 1
    """
    lo, hi = 1, 200
    best = None

    while lo <= hi:
        mid = (lo + hi) // 2
        lines = wrap_text(text, font_name, mid, bbox_w)
        total_h = len(lines) * (mid * 1.2)  # 1.2 * font_size = line height

        if total_h <= bbox_h:
            best = (mid, lines)
            lo = mid + 1
        else:
            hi = mid - 1

    return best