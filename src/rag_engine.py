"""RAG Engine module for document processing and retrieval."""

import csv
import math
import os
import re
import uuid
from typing import Dict, List, Tuple

import chromadb
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
from sentence_transformers import CrossEncoder, SentenceTransformer


class RAGEngine:
    """Enhanced RAG Engine with PDF processing and two-stage retrieval."""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "docs_collection",
    ):
        """Initialize the Enhanced RAG Engine.

        Args:
            db_path: Path to ChromaDB persistent storage.
            collection_name: Name of the ChromaDB collection.
        """
        # 1. Initialize Vector DB (ChromaDB)
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        # 2. Create or get a collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )

        # 3. Initialize Embedding Model (all-MiniLM-L6-v2)
        print("Loading embedding model... (this happens only once)")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # 4. Initialize Reranker (learned from LightRAG)
        # CrossEncoder is slower but more accurate for final scoring
        print("Loading reranker model...")
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        print("Models loaded.")

    def _detect_content_type(self, text: str) -> str:
        """Detect if text contains equations, code, or tables.

        Args:
            text: The text content to analyze.

        Returns:
            Content type: 'equation', 'code', 'table', or 'text'.
        """
        if self._is_equation(text):
            return "equation"
        if self._is_code(text):
            return "code"
        if self._is_table(text):
            return "table"
        return "text"

    @staticmethod
    def _is_equation(text: str) -> bool:
        """Check whether text looks like a math equation."""
        math_indicators = [
            r"\$\$.*?\$\$",
            r"\$.*?\$",
            r"\\text\{",
            r"\\frac\{",
            r"\\sum",
            r"\\int",
            r"\\left",
            r"\\right",
            r"[\u2211\u222b\u2202\u2207\u2248\u2260\u2264\u2265\u00b1\u00d7\u00f7]",
        ]
        math_count = sum(
            1 for p in math_indicators if re.search(p, text)
        )
        return math_count >= 2

    @staticmethod
    def _is_code(text: str) -> bool:
        """Check whether text looks like source code."""
        if re.search(r"^\s*(```|~~~)", text, re.MULTILINE):
            return True
        if re.search(r"</?(code|pre)\b", text, re.IGNORECASE):
            return True

        code_patterns = [
            r"^\s{4,}\w+",
            r"\bdef\s+\w+\s*\(",
            r"\bclass\s+\w+",
            r"\bimport\s+\w+",
            r"\breturn\s+",
            r"^\w+\s*=\s*\w+\(",
            r"[{};]\s*$",
        ]
        code_count = sum(
            1
            for p in code_patterns
            if re.search(p, text, re.MULTILINE)
        )
        return code_count >= 2

    @staticmethod
    def _is_table(text: str) -> bool:
        """Check whether text looks like a table."""
        if re.search(r"^\s*\|.*\|\s*$", text, re.MULTILINE):
            if re.search(
                r"^\s*\|?\s*:?-{3,}\s*\|", text, re.MULTILINE
            ):
                return True
            if text.count("|") >= 4:
                return True

        lines = [
            line for line in text.splitlines() if line.strip()
        ]
        comma_lines = [
            line for line in lines if line.count(",") >= 2
        ]
        if len(comma_lines) >= 2:
            return True

        return text.count("\t") > 3

    # ------------------------------------------------------------------
    # File-reading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_text_file(file_path: str) -> str:
        """Read a text-based file with encoding fallbacks.

        Args:
            file_path: Path to the text file.

        Returns:
            The file content as a string.
        """
        encodings = ["utf-8", "utf-8-sig", "latin-1"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as fh:
                    return fh.read()
            except UnicodeDecodeError:
                continue
        with open(
            file_path, "r", encoding="utf-8", errors="replace"
        ) as fh:
            return fh.read()

    def _extract_text_to_chunks(
        self, text: str, source: str, page: int = 1
    ) -> List[Dict]:
        """Convert raw text into chunk dicts with metadata.

        Args:
            text: Raw text to process.
            source: Source file name.
            page: Page number for the metadata.

        Returns:
            List of chunk dictionaries.
        """
        if not text or not text.strip():
            return []

        chunks: List[Dict] = []
        paragraphs = re.split(r"\n\n+", text)

        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:
                continue
            content_type = self._detect_content_type(para)
            chunks.append(
                {
                    "text": para,
                    "page": page,
                    "type": content_type,
                    "source": source,
                }
            )
        return chunks

    @staticmethod
    def _split_table_rows(
        rows: List[List[str]], max_rows: int = 50
    ) -> List[List[List[str]]]:
        """Split large tables into smaller chunks.

        Uses the first row as a repeated header when splitting.

        Args:
            rows: Table rows.
            max_rows: Maximum rows per chunk.

        Returns:
            List of table chunks.
        """
        if not rows:
            return []
        if len(rows) <= max_rows:
            return [rows]

        header = rows[0]
        body = rows[1:]
        table_chunks: List[List[List[str]]] = []
        chunk_body_size = max_rows - 1

        for i in range(0, len(body), chunk_body_size):
            table_chunks.append(
                [header] + body[i : i + chunk_body_size]
            )

        return table_chunks

    # ------------------------------------------------------------------
    # Per-format extractors
    # ------------------------------------------------------------------

    def _extract_with_layout(self, file_path: str) -> List[Dict]:
        """Extract text from PDF with layout preservation.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of chunk dicts with metadata.
        """
        chunks: List[Dict] = []
        source = os.path.basename(file_path)

        with fitz.open(file_path) as doc:
            for page_index, page in enumerate(doc):
                page_num = page_index + 1
                text = page.get_text("text", sort=True)
                chunks.extend(
                    self._extract_text_to_chunks(
                        text, source, page_num
                    )
                )

                tables = self._extract_tables_pymupdf(page)
                for table in tables:
                    for trows in self._split_table_rows(table):
                        if trows:
                            ttext = self._format_table(trows)
                            chunks.append(
                                {
                                    "text": f"[TABLE]\n{ttext}",
                                    "page": page_num,
                                    "type": "table",
                                    "source": source,
                                }
                            )

        return chunks

    @staticmethod
    def _extract_tables_pymupdf(page) -> List[List[List[str]]]:
        """Best-effort table extraction with PyMuPDF.

        Args:
            page: A PyMuPDF page object.

        Returns:
            List of tables, each table being a list of rows.
        """
        if not hasattr(page, "find_tables"):
            return []

        try:
            table_finder = page.find_tables()
        except Exception:  # pylint: disable=broad-except
            return []

        if not table_finder or not getattr(
            table_finder, "tables", None
        ):
            return []

        tables: List[List[List[str]]] = []
        for table in table_finder.tables:
            try:
                table_rows = table.extract()
            except Exception:  # pylint: disable=broad-except
                continue
            if table_rows:
                tables.append(table_rows)

        return tables

    def _extract_text_file(self, file_path: str) -> List[Dict]:
        """Extract chunks from a plain-text or Markdown file."""
        text = self._read_text_file(file_path)
        source = os.path.basename(file_path)
        return self._extract_text_to_chunks(text, source, page=1)

    def _extract_docx(self, file_path: str) -> List[Dict]:
        """Extract chunks from a DOCX file."""
        doc = Document(file_path)
        source = os.path.basename(file_path)

        paragraphs = [
            p.text.strip()
            for p in doc.paragraphs
            if p.text and p.text.strip()
        ]
        text = "\n\n".join(paragraphs)
        chunks = self._extract_text_to_chunks(
            text, source, page=1
        )

        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [
                    cell.text.strip() if cell.text else ""
                    for cell in row.cells
                ]
                rows.append(cells)

            for trows in self._split_table_rows(rows):
                if trows:
                    ttext = self._format_table(trows)
                    chunks.append(
                        {
                            "text": f"[TABLE]\n{ttext}",
                            "page": 1,
                            "type": "table",
                            "source": source,
                        }
                    )

        return chunks

    def _extract_xlsx(self, file_path: str) -> List[Dict]:
        """Extract chunks from an Excel file."""
        workbook = load_workbook(
            file_path, data_only=True, read_only=True
        )
        source = os.path.basename(file_path)
        chunks: List[Dict] = []

        for sheet_idx, sheet in enumerate(workbook.worksheets, 1):
            rows: List[List[str]] = []
            for row in sheet.iter_rows(values_only=True):
                clean = [
                    "" if c is None else str(c).strip()
                    for c in row
                ]
                if any(cell for cell in clean):
                    rows.append(clean)

            for trows in self._split_table_rows(rows):
                if trows:
                    ttext = self._format_table(trows)
                    chunks.append(
                        {
                            "text": (
                                f"[TABLE]\nSheet: {sheet.title}"
                                f"\n{ttext}"
                            ),
                            "page": sheet_idx,
                            "type": "table",
                            "source": source,
                        }
                    )

        return chunks

    def _extract_csv(self, file_path: str) -> List[Dict]:
        """Extract chunks from a CSV file."""
        encodings = ["utf-8-sig", "utf-8", "latin-1"]
        rows: List[List[str]] = []

        for enc in encodings:
            try:
                with open(
                    file_path, "r", newline="", encoding=enc
                ) as fh:
                    reader = csv.reader(fh)
                    for row in reader:
                        clean = [c.strip() for c in row]
                        if any(clean):
                            rows.append(clean)
                break
            except UnicodeDecodeError:
                continue

        if not rows:
            with open(
                file_path,
                "r",
                newline="",
                encoding="utf-8",
                errors="replace",
            ) as fh:
                reader = csv.reader(fh)
                for row in reader:
                    clean = [c.strip() for c in row]
                    if any(clean):
                        rows.append(clean)

        source = os.path.basename(file_path)
        chunks: List[Dict] = []

        for trows in self._split_table_rows(rows):
            if trows:
                ttext = self._format_table(trows)
                chunks.append(
                    {
                        "text": f"[TABLE]\n{ttext}",
                        "page": 1,
                        "type": "table",
                        "source": source,
                    }
                )

        return chunks

    def _extract_html(self, file_path: str) -> List[Dict]:
        """Extract chunks from an HTML file."""
        html = self._read_text_file(file_path)
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{2,}", "\n\n", text).strip()

        source = os.path.basename(file_path)
        return self._extract_text_to_chunks(
            text, source, page=1
        )

    def _extract_pptx(self, file_path: str) -> List[Dict]:
        """Extract chunks from a PowerPoint file."""
        prs = Presentation(file_path)
        source = os.path.basename(file_path)
        chunks: List[Dict] = []

        for slide_idx, slide in enumerate(prs.slides, 1):
            text_parts: List[str] = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text_parts.append(shape.text)

                if getattr(shape, "has_table", False):
                    rows = []
                    for row in shape.table.rows:
                        cells = [
                            cell.text.strip()
                            if cell.text
                            else ""
                            for cell in row.cells
                        ]
                        rows.append(cells)

                    for trows in self._split_table_rows(rows):
                        if trows:
                            ttext = self._format_table(trows)
                            chunks.append(
                                {
                                    "text": f"[TABLE]\n{ttext}",
                                    "page": slide_idx,
                                    "type": "table",
                                    "source": source,
                                }
                            )

            slide_text = "\n\n".join(
                part.strip()
                for part in text_parts
                if part and part.strip()
            )
            chunks.extend(
                self._extract_text_to_chunks(
                    slide_text, source, page=slide_idx
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Formatting / chunking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_table(table: List[List]) -> str:
        """Format table data as a markdown table string.

        Args:
            table: Table data as a list of rows.

        Returns:
            Markdown-formatted table string.
        """
        if not table:
            return ""

        lines = []
        for row in table:
            clean = [str(c) if c else "" for c in row]
            lines.append("| " + " | ".join(clean) + " |")

        if len(lines) > 1:
            num_cols = len(table[0])
            sep = "| " + " | ".join(["---"] * num_cols) + " |"
            lines.insert(1, sep)

        return "\n".join(lines)

    def _smart_chunk(
        self, chunks: List[Dict], max_chunk_size: int = 1000
    ) -> List[Dict]:
        """Smart chunking that respects content boundaries.

        Args:
            chunks: List of raw extracted chunks.
            max_chunk_size: Maximum character count per chunk.

        Returns:
            List of processed chunks with metadata.
        """
        smart_chunks: List[Dict] = []
        cur: Dict = {"text": "", "metadata": {}}

        for chunk in chunks:
            chunk_text = chunk["text"]

            if chunk["type"] in ["table", "code", "equation"]:
                if cur["text"]:
                    smart_chunks.append(cur)
                    cur = {"text": "", "metadata": {}}

                smart_chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": self._build_meta(
                            chunk, chunk_text
                        ),
                    }
                )
                continue

            combined = len(cur["text"]) + len(chunk_text)
            if combined < max_chunk_size:
                if cur["text"]:
                    cur["text"] += "\n\n" + chunk_text
                else:
                    cur["text"] = chunk_text
                cur["metadata"] = self._build_meta(
                    chunk, cur["text"]
                )
            else:
                if cur["text"]:
                    smart_chunks.append(cur)
                cur = {
                    "text": chunk_text,
                    "metadata": self._build_meta(
                        chunk, chunk_text
                    ),
                }

        if cur["text"]:
            smart_chunks.append(cur)

        return smart_chunks

    @staticmethod
    def _build_meta(chunk: Dict, text: str) -> Dict:
        """Build metadata dict for a chunk.

        Args:
            chunk: Original chunk dict with page, type, source.
            text: The actual text content.

        Returns:
            Metadata dictionary.
        """
        return {
            "page": chunk["page"],
            "type": chunk.get("type", "text"),
            "source": chunk["source"],
            "char_count": len(text),
            "token_count": len(text) // 4,
            "char_count_raw": len(text),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str) -> str:
        """Read a supported file and store chunks in ChromaDB.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            Success message with the number of chunks processed.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        ext = os.path.splitext(file_path)[1].lower()
        extractors = {
            ".pdf": self._extract_with_layout,
            ".txt": self._extract_text_file,
            ".md": self._extract_text_file,
            ".docx": self._extract_docx,
            ".xlsx": self._extract_xlsx,
            ".xlsm": self._extract_xlsx,
            ".csv": self._extract_csv,
            ".html": self._extract_html,
            ".htm": self._extract_html,
            ".pptx": self._extract_pptx,
        }

        extractor = extractors.get(ext)
        if extractor is None:
            raise ValueError(f"Unsupported file type: {ext}")

        raw_chunks = extractor(file_path)
        processed = self._smart_chunk(raw_chunks)

        if not processed:
            basename = os.path.basename(file_path)
            return f"No extractable content found in {basename}"

        texts = [c["text"] for c in processed]
        metadatas = [c["metadata"] for c in processed]
        batch_id = uuid.uuid4().hex[:8]
        ids = [
            f"{batch_id}_chunk_{i}" for i in range(len(texts))
        ]

        embeddings = self.embedding_model.encode(texts).tolist()

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        basename = os.path.basename(file_path)
        return (
            f"Successfully processed {len(texts)} chunks"
            f" from {basename}"
        )

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> Tuple[List[str], List[Dict]]:
        """Retrieve the most relevant text chunks.

        Uses two-stage retrieval: vector search then
        cross-encoder reranking.

        Args:
            query: The search query string.
            top_k: Number of top results to return.

        Returns:
            Tuple of (documents, metadatas) with confidence.
        """
        initial_k = top_k * 3
        query_emb = self.embedding_model.encode(
            [query]
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=initial_k,
            include=["documents", "metadatas", "distances"],
        )

        if (
            not results
            or not results["documents"]
            or not results["documents"][0]
        ):
            return [], []

        flat_docs = results["documents"][0]
        flat_metas = (
            results["metadatas"][0]
            if results.get("metadatas")
            else [{}] * len(flat_docs)
        )

        pairs = [[query, doc] for doc in flat_docs]
        scores = self.reranker.predict(pairs)

        scored = self._score_results(
            flat_docs, flat_metas, scores
        )
        scored.sort(key=lambda x: x["score"], reverse=True)
        final = scored[:top_k]

        return (
            [item["doc"] for item in final],
            [item["meta"] for item in final],
        )

    @staticmethod
    def _score_results(
        docs: List[str],
        metas: List[Dict],
        scores,
    ) -> List[Dict]:
        """Combine documents, metadata, and reranker scores.

        Args:
            docs: Retrieved document texts.
            metas: Retrieved metadata dicts.
            scores: Raw reranker scores.

        Returns:
            List of scored result dicts.
        """
        scored: List[Dict] = []
        for doc, meta, score in zip(docs, metas, scores):
            confidence = 1 / (1 + math.exp(-score))
            enriched = dict(meta) if meta else {}
            enriched["confidence"] = round(confidence, 3)
            enriched["rerank_score"] = float(score)
            scored.append(
                {
                    "doc": doc,
                    "meta": enriched,
                    "score": score,
                }
            )
        return scored

    def analyze_query(self, query: str) -> dict:
        """Analyze query for search terms and rewrite.

        Args:
            query: The user query string.

        Returns:
            Dict with query_rewrite, search_terms, and
            retrieval_strategy keys.
        """
        stopwords = {
            "what", "is", "the", "a", "an", "how", "does",
            "do", "can", "could", "would", "should", "will",
            "are", "was", "were", "been", "being", "have",
            "has", "had", "of", "to", "for", "in", "on",
            "with", "by", "about", "this", "that", "these",
            "those", "it", "its", "and", "or",
        }

        words = re.findall(r"\b\w+\b", query.lower())
        search_terms = [
            w
            for w in words
            if w not in stopwords and len(w) > 2
        ]

        query_rewrite = query.strip()
        if query_rewrite.endswith("?"):
            query_rewrite = query_rewrite[:-1]

        strategy = (
            "Semantic search using local embeddings"
            " (all-MiniLM-L6-v2)"
        )
        return {
            "query_rewrite": query_rewrite,
            "search_terms": search_terms[:6],
            "retrieval_strategy": strategy,
        }