import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import os
import re
import config

# Make sure pytesseract points to your Tesseract install
# Update the path if your installation is different.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class DocumentProcessor:
    """
    Handles multi-modal extraction from a PDF:
    - Text (chunked)
    - Tables (numeric-heavy blocks)
    - Images with OCR
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

        # Ensure image output directory exists
        os.makedirs(config.IMAGES_DIR, exist_ok=True)

    def close(self):
        if self.doc is not None:
            self.doc.close()
            self.doc = None

    # ---------- helpers ----------

    def _chunk_text(self, text: str, max_chars: int = 1000):
        """
        Split raw page text into smaller, semantic chunks
        based on blank lines + length.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks = []
        current = ""

        for p in paragraphs:
            if len(current) + len(p) + 2 > max_chars:
                if current:
                    chunks.append(current.strip())
                current = p
            else:
                current = current + "\n\n" + p if current else p

        if current:
            chunks.append(current.strip())

        return chunks

    def _looks_like_table(self, text: str, min_numbers: int = 12, min_ratio: float = 0.15):
        """
        Heuristic: consider a block a table if:
        - It has at least `min_numbers` numeric tokens, AND
        - Numbers are at least `min_ratio` of all tokens.
        """
        nums = re.findall(r"\d+(\.\d+)?", text)
        if len(nums) < min_numbers:
            return False

        tokens = text.split()
        if not tokens:
            return False

        ratio = len(nums) / len(tokens)
        return ratio >= min_ratio

    # ---------- extraction ----------

    def extract_text_chunks(self):
        chunks = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()

            if text.strip():
                sub_chunks = self._chunk_text(text, max_chars=1000)
                for i, sub in enumerate(sub_chunks):
                    chunks.append(
                        {
                            "type": "text",
                            "content": sub,
                            "page": page_num + 1,
                            "source": f"Page {page_num + 1}, chunk {i + 1}",
                        }
                    )
        return chunks

    def extract_tables(self):
        tables = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    lines = block["lines"]
                    if len(lines) > 2:
                        table_text = ""
                        for line in lines:
                            for span in line["spans"]:
                                table_text += span["text"] + " "
                            table_text += "\n"
                        table_text = table_text.strip()

                        if table_text and self._looks_like_table(table_text):
                            tables.append(
                                {
                                    "type": "table",
                                    "content": table_text,
                                    "page": page_num + 1,
                                    "source": f"Table on Page {page_num + 1}",
                                }
                            )
        return tables

    def extract_images_with_ocr(self):
        images_data = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]

                try:
                    pix = page.get_pixmap(xref=xref)
                    image_bytes = pix.tobytes("png")
                    img_pil = Image.open(io.BytesIO(image_bytes))

                    # OCR
                    ocr_text = pytesseract.image_to_string(img_pil)
                    if not ocr_text.strip():
                        continue

                    # Save image to disk
                    image_filename = os.path.join(
                        config.IMAGES_DIR, f"page{page_num + 1}_img{img_index + 1}.png"
                    )
                    img_pil.save(image_filename)

                    first_line = ocr_text.strip().split("\n")[0][:80]
                    images_data.append(
                        {
                            "type": "image",
                            "content": ocr_text,
                            "page": page_num + 1,
                            "image_path": image_filename,
                            "source": f"Image on Page {page_num + 1}: {first_line}",
                        }
                    )
                except Exception as e:
                    print(f"OCR failed on page {page_num + 1}: {e}")

        return images_data

    # ---------- main pipeline ----------

    def process_document(self):
        print(f"Processing document: {self.pdf_path}")

        text_chunks = self.extract_text_chunks()
        tables = self.extract_tables()
        images = self.extract_images_with_ocr()

        print(f"Extracted {len(text_chunks)} text chunks")
        print(f"Extracted {len(tables)} tables")
        print(f"Extracted {len(images)} images with OCR")
        print(f" Total chunks: {len(text_chunks) + len(tables) + len(images)}")

        chunks = text_chunks + tables + images

        return chunks


if __name__ == "__main__":
    processor = DocumentProcessor("qatar_test_doc.pdf")
    chunks = processor.process_document()
    print(f"\nSample chunk: {chunks[0]}")
    processor.close()