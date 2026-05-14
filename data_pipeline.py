import re
import json
import random
import os
import hashlib
import nltk
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from tqdm import tqdm

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# SABİTLER

@dataclass
class PipelineConfig:
    # Pencere ayarları
    window_size: int = 20
    stride:      int = 10

    # Cümle filtresi
    min_sent_len: int = 10
    max_sent_len: int = 600

    # Makale limitleri
    wiki_en_articles:       int = 2000
    wiki_tr_articles:       int = 1500
    openwebtext_articles:   int = 500

    # Çıktı
    output_dir:  str   = "data/"
    train_ratio: float = 0.80
    val_ratio:   float = 0.10


CFG = PipelineConfig()


# 1. VERİ YAPISI

@dataclass
class Segment:
    sentences:  List[str]
    boundaries: List[int]
    source:     str
    language:   str
    doc_id:     str

    def __post_init__(self):
        assert len(self.sentences) == len(self.boundaries), (
            f"sentences ({len(self.sentences)}) != boundaries ({len(self.boundaries)})"
        )


# 2. YARDIMCI FONKSİYONLAR

def make_doc_id(prefix: str, title: str) -> str:
    
    # Çakışmaya karşı MD5 hash tabanlı doc_id
    
    slug  = re.sub(r'[^a-z0-9]', '_', title.lower())[:30]
    h     = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{slug}_{h}"


def split_sentences(text: str, language: str = "en") -> List[str]:

    lang      = "turkish" if language == "tr" else "english"
    sentences = nltk.sent_tokenize(text, language=lang)
    return [
        s.strip() for s in sentences
        if CFG.min_sent_len < len(s.strip()) < CFG.max_sent_len
    ]


def detect_language(text: str) -> str:

    tr_chars = set("çğışöüÇĞİŞÖÜ")
    tr_count = sum(1 for c in text if c in tr_chars)
    ratio    = tr_count / max(len(text), 1)
    return "tr" if ratio > 0.005 else "en"


def build_segments(
    sections: List[Tuple[str, str]],
    source:   str,
    language: str,
    doc_id:   str,
) -> List[Segment]:

    all_sentences:  List[str] = []
    all_boundaries: List[int] = []

    valid_sections = [(t, s) for t, s in sections if s.strip()]
    n_sections     = len(valid_sections)

    for sec_idx, (_, text) in enumerate(valid_sections):
        sentences = split_sentences(text, language)
        if not sentences:
            continue

        is_last_section = (sec_idx == n_sections - 1)
        n = len(sentences)

        for sent_idx, sent in enumerate(sentences):
            all_sentences.append(sent)
            is_last_in_section = (sent_idx == n - 1)

            boundary = 1 if (is_last_in_section and not is_last_section) else 0
            all_boundaries.append(boundary)

    if len(all_sentences) < 4:
        return []

    segments: List[Segment] = []
    total    = len(all_sentences)
    win      = CFG.window_size
    step     = CFG.stride

    for start in range(0, total - win + 1, step):
        end = start + win
        segments.append(Segment(
            sentences  = all_sentences[start:end],
            boundaries = all_boundaries[start:end],
            source     = source,
            language   = language,
            doc_id     = doc_id,
        ))

    if total > win and (total - win) % step != 0:
        segments.append(Segment(
            sentences  = all_sentences[-win:],
            boundaries = all_boundaries[-win:],
            source     = source,
            language   = language,
            doc_id     = doc_id,
        ))

    return segments


# WIKIPEDIA

SKIP_SECTIONS = {
    # İngilizce
    "references", "see also", "external links", "notes",
    "bibliography", "further reading", "footnotes",
    "citations", "sources", "works cited",
    # Türkçe
    "kaynakça", "ayrıca bakınız", "dış bağlantılar", "notlar",
    "dipnotlar", "kaynaklar", "bibliyografya",
}

# Wikipedia wikitext  == Başlık == şeklinde
SECTION_PATTERN = re.compile(r'\n={2,4}\s*(.+?)\s*={2,4}\n')


def parse_wikipedia_sections(text: str) -> List[Tuple[str, str]]:

    sections: List[Tuple[str, str]] = []

    parts = SECTION_PATTERN.split(text)

    intro = parts[0].strip()
    if len(intro) > 80:
        sections.append(("__intro__", intro))

    # Başlık-içerik çiftleri
    i = 1
    while i + 1 < len(parts):
        title   = parts[i].strip()
        content = parts[i + 1].strip()
        i += 2

        if title.lower() in SKIP_SECTIONS:
            continue
        if len(content) < 30:
            continue

        sections.append((title, content))

    return sections


def collect_wikipedia(language: str, max_articles: int) -> List[Segment]:
    from datasets import load_dataset

    config = "20231101.tr" if language == "tr" else "20231101.en"
    source = f"wikipedia_{language}"

    dataset  = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    segments: List[Segment] = []
    seen_ids = set()
    pbar     = tqdm(total=max_articles, desc=f"Wikipedia {language.upper()}")

    for article in dataset:
        if pbar.n >= max_articles:
            break
        if len(article["text"]) < 200:
            continue

        sections = parse_wikipedia_sections(article["text"])
        if len(sections) < 2:
            continue

        doc_id = make_doc_id(f"wiki_{language}", article["title"])
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        segs = build_segments(sections, source=source, language=language, doc_id=doc_id)
        if segs:
            segments.extend(segs)
            pbar.update(1)

    pbar.close()
    print(f"[Wikipedia-{language.upper()}] {len(segments)} segment toplandı.")
    return segments


# OPENWEBTEXT (EN)

def collect_openwebtext(max_articles: int) -> List[Segment]:

    from datasets import load_dataset

    source = "openwebtext_en"
  
    try:
        dataset = load_dataset(
            "Skylion007/openwebtext",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"[OpenWebText Yüklenemedi: {e}")
        return []

    segments: List[Segment] = []
    seen_ids: set           = set()
    pbar      = tqdm(total=max_articles, desc="OpenWebText")

    for article in dataset:
        if pbar.n >= max_articles:
            break

        text = (article.get("text") or "").strip()
        if len(text) < 500:
            continue

        # her 4 paragraf bir bölüm
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
        if len(paragraphs) < 4:
            # \n\n yoksa \n ile dene
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 80]
        if len(paragraphs) < 4:
            continue

        sections: List[Tuple[str, str]] = []
        group_size = 4
        for i in range(0, len(paragraphs), group_size):
            group = " ".join(paragraphs[i:i + group_size])
            if len(group) > 100:
                sections.append((f"para_{i}", group))

        if len(sections) < 2:
            continue

        doc_id = make_doc_id("openwebtext_en", text[:80])
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        segs = build_segments(sections, source=source, language="en", doc_id=doc_id)
        if segs:
            segments.extend(segs)
            pbar.update(1)

    pbar.close()
    print(f"[OpenWebText-EN] {len(segments)} segment toplandı.")
    return segments


# KAYDET

def save_dataset(segments: List[Segment], output_dir: str):

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Kaynağa göre
    source_groups: dict = {}
    for seg in segments:
        source_groups.setdefault(seg.source, set()).add(seg.doc_id)

    train_ids: set = set()
    val_ids:   set = set()
    test_ids:  set = set()

    for source, ids in source_groups.items():
        id_list = list(ids)
        random.shuffle(id_list)
        n = len(id_list)

        t_end = int(n * CFG.train_ratio)
        v_end = int(n * (CFG.train_ratio + CFG.val_ratio))

        train_ids.update(id_list[:t_end])
        val_ids.update(id_list[t_end:v_end])
        test_ids.update(id_list[v_end:])

    splits = {
        "train": [s for s in segments if s.doc_id in train_ids],
        "val":   [s for s in segments if s.doc_id in val_ids],
        "test":  [s for s in segments if s.doc_id in test_ids],
    }

    all_stats = {}

    for name, segs in splits.items():
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for seg in segs:
                f.write(json.dumps(asdict(seg), ensure_ascii=False) + "\n")

        total_sents  = sum(len(s.sentences)  for s in segs)
        total_bounds = sum(sum(s.boundaries) for s in segs)
        ratio        = total_bounds / total_sents if total_sents else 0

        # Kaynak ve dil dağılımı
        sources:   dict = {}
        languages: dict = {}
        for s in segs:
            sources[s.source]     = sources.get(s.source, 0) + 1
            languages[s.language] = languages.get(s.language, 0) + 1

        all_stats[name] = {
            "n_segments":     len(segs),
            "n_sentences":    total_sents,
            "n_boundaries":   total_bounds,
            "boundary_ratio": round(ratio, 6),
            "sources":        sources,
            "languages":      languages,
        }

        print(
            f"[{name.upper():5}] {len(segs):6} segment | "
            f"boundary: {ratio:.2%} | "
            f"dil: {languages} | "
            f"→ {path}"
        )

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\n[Stats] → {stats_path}")



# MAIN

if __name__ == "__main__":
    random.seed(42)

    ALL_SEGMENTS: List[Segment] = []

    ALL_SEGMENTS += collect_wikipedia(language="en", max_articles=CFG.wiki_en_articles)
    ALL_SEGMENTS += collect_wikipedia(language="tr", max_articles=CFG.wiki_tr_articles)

    ALL_SEGMENTS += collect_openwebtext(max_articles=CFG.openwebtext_articles)

    print(f"\n{'─'*50}")
    print(f" Toplam {len(ALL_SEGMENTS)} segment")

    # Kaynak ve dil dağılımı
    source_counts: dict = {}
    lang_counts:   dict = {}
    for s in ALL_SEGMENTS:
        source_counts[s.source]   = source_counts.get(s.source, 0) + 1
        lang_counts[s.language]   = lang_counts.get(s.language, 0) + 1

    print("\nKaynak dağılımı:")
    for src, cnt in sorted(source_counts.items()):
        pct = cnt / len(ALL_SEGMENTS) * 100
        print(f"  {src:<25} {cnt:6} segment  ({pct:.1f}%)")

    print(f"\nDil dağılımı: {lang_counts}")
    print(f"{'─'*50}\n")

    save_dataset(ALL_SEGMENTS, output_dir=CFG.output_dir)
    print("\n Bitti.")
