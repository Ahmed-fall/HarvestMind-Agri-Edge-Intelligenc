import fitz
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Configure production logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HarvestMindChunker")

# Resolve project paths dynamically
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"

# ==========================================
# 1. DATA MODELS & CONFIGURATION
# ==========================================
@dataclass
class TransformationRule:
    target_section: str
    marker: str
    new_section_name: str
    page_offset: int = 0
    clean_infographic_digits: bool = False  # Flag to clean stray digits from layout

@dataclass
class DocumentProfile:
    doc_id: str
    filename: str
    skip_pages: List[int] = field(default_factory=list)
    content_end_page: Optional[int] = None
    page_offset: int = 0
    min_word_count: int = 40
    max_word_count: int = 350
    overlap_words: int = 50
    ordered_headings: List[str] = field(default_factory=list)
    noise_patterns: List[re.Pattern] = field(default_factory=list)
    transformations: List[TransformationRule] = field(default_factory=list)

# ==========================================
# 2. CORE ENGINES
# ==========================================
class LayoutAnalyzer:
    @staticmethod
    def extract_blocks(page: fitz.Page, profile: DocumentProfile) -> List[str]:
        raw_blocks = page.get_text("blocks")
        if not raw_blocks:
            return []
            
        valid_blocks = []
        for b in raw_blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if block_type != 0:
                continue
            text = text.strip()
            if not text or any(p.match(text) for p in profile.noise_patterns):
                continue
            valid_blocks.append((x0, y0, x1, y1, text))

        if not valid_blocks:
            return []

        # Horizontal Banding Strategy
        valid_blocks.sort(key=lambda b: b[1])
        bands = []
        for b in valid_blocks:
            x0, y0, x1, y1, text = b
            matched_band = False
            for band in bands:
                band_y0, band_y1 = band["y_range"]
                overlap = max(0, min(y1, band_y1) - max(y0, band_y0))
                block_height = y1 - y0
                
                if overlap > 0.1 * block_height:
                    band["blocks"].append(b)
                    band["y_range"] = (min(band_y0, y0), max(band_y1, y1))
                    matched_band = True
                    break
                    
            if not matched_band:
                bands.append({"y_range": (y0, y1), "blocks": [b]})

        bands.sort(key=lambda band: band["y_range"][0])
        ordered_texts = []
        page_width = page.rect.width
        mid_x = page_width / 2

        for band in bands:
            left_col, right_col = [], []
            for b in band["blocks"]:
                if b[0] < mid_x:
                    left_col.append(b)
                else:
                    right_col.append(b)
                    
            left_col.sort(key=lambda b: b[1])
            right_col.sort(key=lambda b: b[1])
            
            ordered_texts.extend([b[4] for b in left_col])
            ordered_texts.extend([b[4] for b in right_col])

        return ordered_texts

class DocumentParser:
    def __init__(self, profile: DocumentProfile):
        self.profile = profile
        self.headings = [h.strip() for h in profile.ordered_headings]
        self.normalized_headings = [self._normalize(h) for h in self.headings]

    def _normalize(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip().lower()

    def _detect_and_split_heading(self, text: str) -> tuple:
        """
        Returns (detected_heading_original_name, remaining_text)
        If no heading is detected, returns (None, text)
        """
        if not self.headings:
            return None, text
            
        norm_text = self._normalize(text)
        
        # Check from longest heading to shortest to prevent partial matches
        sorted_indices = sorted(range(len(self.headings)), key=lambda idx: len(self.normalized_headings[idx]), reverse=True)
        
        for idx in sorted_indices:
            h_orig = self.headings[idx]
            h_norm = self.normalized_headings[idx]
            
            # Scenario 1: The block is exactly the heading (allowing for whitespace differences)
            if norm_text == h_norm:
                return h_orig, ""
                
            # Scenario 2: The block starts with the heading followed by a newline or space separation
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                temp_line = ""
                for line_idx, line in enumerate(lines):
                    temp_line = self._normalize(temp_line + " " + line)
                    if temp_line == h_norm:
                        remaining_lines = lines[line_idx + 1:]
                        remaining_text = "\n".join(remaining_lines).strip()
                        return h_orig, remaining_text
                    if len(temp_line) > len(h_norm) + 5:
                        break
                        
        return None, text

    def _clean(self, text: str) -> str:
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
        text = re.sub(r'^([A-Z])\s+([a-z])', lambda m: m.group(1) + m.group(2), text)
        return re.sub(r'\s+', ' ', text).strip()

    def process(self, pdf_path: Path) -> List[Dict]:
        doc = fitz.open(str(pdf_path))
        raw_chunks = []
        
        curr_section = "Introduction"
        curr_page = (self.profile.skip_pages[-1] + 1 + self.profile.page_offset) if self.profile.skip_pages else 1
        curr_blocks = []
        end_idx = self.profile.content_end_page if self.profile.content_end_page else len(doc)

        for i in range(len(doc)):
            if i in self.profile.skip_pages or i >= end_idx: continue
                
            printed_page = i + self.profile.page_offset
            blocks = LayoutAnalyzer.extract_blocks(doc[i], self.profile)
            
            for text in blocks:
                heading, remaining = self._detect_and_split_heading(text)
                if heading:
                    self._flush(raw_chunks, curr_section, curr_page, curr_blocks, pdf_path.name)
                    curr_section = heading
                    curr_page = printed_page
                    curr_blocks = []
                    if remaining:
                        curr_blocks.append(remaining)
                else:
                    curr_blocks.append(text)
                    
        self._flush(raw_chunks, curr_section, curr_page, curr_blocks, pdf_path.name)
        
        transformed_chunks = self._apply_transformations(raw_chunks)
        final_chunks = self._enforce_token_limits(transformed_chunks)
        
        for i, c in enumerate(final_chunks):
            c['chunk_id'] = f"{self.profile.doc_id}_{i+1:03d}"
            
        return final_chunks

    def _flush(self, chunks: list, section: str, page: int, blocks: list, filename: str):
        if not blocks: return
        text = self._clean(' '.join(blocks))
        if len(text.split()) >= self.profile.min_word_count:
            chunks.append({
                "doc_id": self.profile.doc_id,
                "source": filename,
                "section": section,
                "page": page,
                "text": text,
                "word_count": len(text.split())
            })

    def _apply_transformations(self, chunks: List[Dict]) -> List[Dict]:
        if not self.profile.transformations: return chunks
            
        final = []
        for chunk in chunks:
            processed = False
            for rule in self.profile.transformations:
                if chunk['section'].startswith(rule.target_section) and rule.marker in chunk['text']:
                    text = chunk['text']
                    idx = text.find(rule.marker)
                    if idx > 0:
                        part_a, part_b = text[:idx].strip(), text[idx:].strip()
                        
                        if rule.clean_infographic_digits:
                            part_b = re.sub(r'\b[1-9]\b', '', part_b)
                            part_b = re.sub(r'\s+', ' ', part_b).strip()

                        if len(part_a.split()) >= self.profile.min_word_count:
                            chunk['text'] = part_a
                            chunk['word_count'] = len(part_a.split())
                            final.append(chunk)
                        
                        if len(part_b.split()) >= self.profile.min_word_count:
                            final.append({
                                "doc_id": chunk['doc_id'],
                                "source": chunk['source'],
                                "section": rule.new_section_name,
                                "page": chunk['page'] + rule.page_offset,
                                "text": part_b,
                                "word_count": len(part_b.split())
                            })
                        processed = True
                        break
            if not processed: 
                final.append(chunk)
                
        return final

    def _enforce_token_limits(self, chunks: List[Dict]) -> List[Dict]:
        final = []
        for chunk in chunks:
            word_count = chunk['word_count']
            if word_count <= self.profile.max_word_count:
                final.append(chunk)
            else:
                words = chunk['text'].split()
                start = 0
                part_idx = 1
                while start < word_count:
                    end = min(start + self.profile.max_word_count, word_count)
                    chunk_words = words[start:end]
                    
                    final.append({
                        "doc_id": chunk['doc_id'],
                        "source": chunk['source'],
                        "section": f"{chunk['section']} (Part {part_idx})",
                        "page": chunk['page'],
                        "text": ' '.join(chunk_words),
                        "word_count": len(chunk_words)
                    })
                    start += (self.profile.max_word_count - self.profile.overlap_words)
                    part_idx += 1
        return final

# ==========================================
# 3. FAO HARVESTMIND PROFILE CONFIGURATION
# ==========================================
fao_profile = DocumentProfile(
    doc_id="fao",
    filename="fao-1.pdf",
    skip_pages=[0, 1, 2, 3, 4],
    content_end_page=46,
    page_offset=1,
    min_word_count=40,
    max_word_count=350,
    overlap_words=50,
    ordered_headings=[
        "1. Introduction", "2. Instructions for the Implementation of SmallScale Storage Practices",
        "Physical factors that affect grain in storage", "Determining moisture content",
        "Conditions for seed storage", "Common storage pests", "Primary insect pests",
        "The grain weevil (Sitophilus spp.)", "The larger grain borer (Prostephanus truncatus)",
        "The lesser grain borer (Rhyzopertha dominica)", "The grain moth (Sitotroga cerealella)",
        "The cowpea weevil (Callosobruchus maculatus)", "Secondary insect pests",
        "The red rust flour beetle (Tribolium spp.)", "The tropical warehouse moth (Ephestia spp.)",
        "Mould", "Termites (Macrotermes spp.)", "Rodents", "Birds", "Hygiene",
        "Integrated pest management in the control of storage insects",
        "Required Conditions for Seed Treatment in an FAO-supported intervention",
        "Use of pesticides2", "Safety precautions", "Selection and procurement of pesticides",
        "Pesticide management", "Termite control", "Rodent control", "Non-chemical rodent control",
        "Bird control", "Prestorage handling", "Prestorage handling of rice",
        "Prestorage handling of groundnuts", "Prestorage handling of maize", "Prestorage handling of beans",
        "Small-scale storage facilities", "Traditional storage facilities", "Open storage",
        "Semi–open storage", "Closed storage", "Modern storage facilities", "The storage grain bag",
        "Modern or adapted granaries", "Modern cribs", "Metal silo bins", "Hermetic bags",
        "Insecticide-treated bags", "Small containers", "3. Conclusions"
    ],
    noise_patterns=[
        re.compile(r'^Figure \d+', re.I), re.compile(r'^©'), re.compile(r'^Table \d+', re.I),
        re.compile(r'^\d{1,2}$'), re.compile(r'^FAO Sub-regional'), re.compile(r'^Sub-Regional Coordinator'),
        re.compile(r'^Senior Coordinator'), re.compile(r'^David Phiri$'), re.compile(r'^Mario Samaja$'),
        re.compile(r'^Johannesburg$'), re.compile(r'^Harare$'), re.compile(r'^Southern Africa$'),
        re.compile(r'^\d+\s+Biological control'), re.compile(r'^enemies, antagonists'),
        re.compile(r'^\(IPPC:'), re.compile(r'^\d+\s+The overall framework'),
        re.compile(r'^by the International Code'), re.compile(r'^accompanying technical'),
        re.compile(r'^storage of pesticides.*labelling'), re.compile(r'^pesticide materials'),
        re.compile(r'^theme/pests'), re.compile(r'^ISBN|^E-ISBN|^I3769E'),
        re.compile(r'^Funded by'), re.compile(r'^Coordinator:'), re.compile(r'^Humanitarian Aid'),
        re.compile(r'^Rice growing|^Harvesting rice|^Threshing.*drying|^Groundnuts growing|^Dried maize|^Beans growing|^Grading beans|^Mould on maize|^Termite damage|^Storage with poles|^baffles', re.I),
        re.compile(r'^(right|left).*:'),
        re.compile(r'^\(Prostephanus|^\(Sitophilus|^\(Sitotroga|^\(Rhyzopertha|^\(Callosobruchus|^\(Tribolium|^\(Macrotermes')
    ],
    transformations=[
        TransformationRule(
            target_section="Hermetic bags",
            marker="Place the silo under cover",
            new_section_name="Basic steps for appropriate silo use",
            page_offset=1,
            clean_infographic_digits=True  # Automatically scrubs stray numbering
        )
    ]
)

# ==========================================
# 4. EXECUTION ENTRY POINT
# ==========================================
def main():
    KB_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = KB_DIR / fao_profile.filename
    output_path = KB_DIR / f"{fao_profile.doc_id}_chunks.json"

    if not pdf_path.exists():
        logger.error(f"Target PDF not found at {pdf_path}. Please place the file in data/knowledge_base/")
        return

    logger.info(f"Initializing Chunking Pipeline for {pdf_path.name}...")
    parser = DocumentParser(profile=fao_profile)
    final_chunks = parser.process(pdf_path)
    
    logger.info(f"Generated {len(final_chunks)} chunks.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Successfully saved chunks to {output_path}")

if __name__ == "__main__":
    main()