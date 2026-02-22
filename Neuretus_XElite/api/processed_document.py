import os


class ProcessedDocument:
    def __init__(self, doc_id: str, output_dir: str):
        self.id = doc_id
        self.output_dir = output_dir
    
    def get_pdf(self) -> str:
        return os.path.join(self.output_dir, self.id, "output.pdf")
    
    def get_md(self) -> str:
        md_path = os.path.join(self.output_dir, self.id, "ocr_output", "result.md")
        if os.path.exists(md_path):
            return md_path
        return None
