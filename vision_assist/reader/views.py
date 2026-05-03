import json
import re
from statistics import mean

import cv2
import easyocr
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency fallback
    pytesseract = None

from .models import Document

# EasyOCR reader initialised once and reused for all requests.
# gpu=False is required on Render (no GPU available).
_reader = easyocr.Reader(['en'], gpu=False)

OCR_READ_KWARGS = {
    'detail': 1,
    'paragraph': False,
    'decoder': 'greedy',
    'batch_size': 1,
    'workers': 0,
    'contrast_ths': 0.1,
    'adjust_contrast': 0.5,
    'text_threshold': 0.6,
    'low_text': 0.3,
    'link_threshold': 0.4,
    'mag_ratio': 1.2,
    'canvas_size': 1600,
}

ALPHA_STRAY_DIGIT_RE = re.compile(r'^(?P<word>[A-Za-z]{2,})\s+(?P<digit>\d)$')
ALPHA_ATTACHED_DIGIT_RE = re.compile(r'^(?P<word>[A-Za-z]{2,})(?P<digit>\d)$')

TESSERACT_CONFIG = '--oem 3 --psm 6'
FAST_OCR_SCORE = 78

TECH_PHRASES = {
    ('machine', 'learning'),
    ('deep', 'learning'),
    ('natural', 'language'),
    ('neural', 'network'),
    ('image', 'processing'),
    ('object', 'detection'),
    ('feature', 'extraction'),
    ('text', 'classification'),
    ('speech', 'recognition'),
    ('reinforcement', 'learning'),
    ('supervised', 'learning'),
    ('unsupervised', 'learning'),
    ('data', 'science'),
    ('computer', 'vision'),
}


def _load_cv_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Could not read the image file.')
    return image


def _resize_for_ocr(image, min_width=1200):
    height, width = image.shape[:2]
    if width >= min_width:
        return image
    scale = min_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)


def _prepare_ocr_variants(image):
    base = _resize_for_ocr(image, min_width=1200)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    # Use bilateral filter to remove noise while preserving text edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    return [
        ('base', base),
        ('gray', gray),
        ('denoised', denoised),
    ]


def _clean_ocr_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _score_text_candidate(text, confidences):
    if not text:
        return 0.0

    alpha_ratio = sum(ch.isalnum() or ch.isspace() for ch in text) / max(len(text), 1)
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    punctuation = sum(not ch.isalnum() and not ch.isspace() for ch in text)
    score = (mean(confidences) if confidences else 0.0) * 100
    score += len(text) * 2
    score += alpha_ratio * 24
    if digits == 0:
        score += 8
    if text.isalpha():
        score += 18
    if letters >= 3 and digits == 1:
        score -= 14
    if letters >= 3 and digits > 1:
        score -= 22
    if len(text.split()) == 1 and text.isalpha() and len(text) <= 10:
        score += 10
    score -= punctuation * 5

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) == 1:
        line = lines[0]
        if ALPHA_STRAY_DIGIT_RE.fullmatch(line) or ALPHA_ATTACHED_DIGIT_RE.fullmatch(line):
            score -= 45

    return score


def _normalize_short_label_noise(line, items):
    tokens = line.split()
    if len(tokens) != 2:
        return line

    first, second = tokens
    if not first.isalpha():
        return line

    if len(first) < 2:
        return line

    if second.isdigit() and len(second) == 1:
        digit_conf = items[-1]['confidence'] if items else 0.0
        if digit_conf < 0.85:
            return first

    return line


def _normalize_technical_phrase_order(line):
    tokens = line.split()
    if len(tokens) != 2:
        return line

    left, right = tokens
    if not left.isalpha() or not right.isalpha():
        return line

    left_lower = left.lower()
    right_lower = right.lower()

    if (left_lower, right_lower) in TECH_PHRASES:
        return line

    if (right_lower, left_lower) in TECH_PHRASES:
        return f'{right} {left}'

    return line


def _compose_text_from_results(results):
    if not results:
        return '', 0.0

    items = []
    for bbox, text, confidence in results:
        cleaned = text.strip()
        conf_val = float(confidence or 0.0)
        # Filter out empty text and low confidence predictions to reduce noise
        if not cleaned or conf_val < 0.25:
            continue
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        items.append({
            'text': cleaned,
            'x': min(xs),
            'y': min(ys),
            'h': max(ys) - min(ys),
            'confidence': conf_val,
        })

    if not items:
        return '', 0.0

    items.sort(key=lambda item: (item['y'], item['x']))
    lines = []
    current_line = [items[0]]
    current_y = items[0]['y']
    current_h = max(items[0]['h'], 1)

    for item in items[1:]:
        y_gap = abs(item['y'] - current_y)
        threshold = max(18, int(current_h * 0.75))
        if y_gap > threshold:
            lines.append(current_line)
            current_line = [item]
            current_y = item['y']
            current_h = max(item['h'], 1)
        else:
            current_line.append(item)
            current_y = min(current_y, item['y'])
            current_h = max(current_h, item['h'])

    lines.append(current_line)

    formatted_lines = []
    confidences = []
    for line in lines:
        ordered = sorted(line, key=lambda item: item['x'])
        line_text = ' '.join(item['text'] for item in ordered).strip()
        if line_text:
            line_text = _normalize_short_label_noise(line_text, ordered)
            line_text = _normalize_technical_phrase_order(line_text)
            formatted_lines.append(line_text)
            confidences.extend(item['confidence'] for item in ordered)

    text = _clean_ocr_text('\n'.join(formatted_lines))
    if not text:
        return '', 0.0

    score = _score_text_candidate(text, confidences)
    return text, score


def _easyocr_candidates(image):
    candidates = []
    for index, (_, variant) in enumerate(_prepare_ocr_variants(image)):
        try:
            results = _reader.readtext(variant, **OCR_READ_KWARGS)
            text, score = _compose_text_from_results(results)
            if text:
                candidates.append((text, score))
                if index == 0 and score >= FAST_OCR_SCORE:
                    return candidates
        except Exception:
            continue
    return candidates


def _tesseract_candidates(image):
    if pytesseract is None:
        return []

    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception:
        return []

    try:
        text = pytesseract.image_to_string(
            pil_image,
            config=TESSERACT_CONFIG,
            lang='eng',
        )
        cleaned = _clean_ocr_text(text)
        if cleaned:
            return [(cleaned, 62.0)]
    except Exception:
        return []

    return []


def _select_best_candidate_detail(candidates):
    best_text = ''
    best_score = float('-inf')
    seen = set()

    for text, score in candidates:
        normalized = _clean_ocr_text(text)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        adjusted_score = score + _score_text_candidate(normalized, [])
        if adjusted_score > best_score:
            best_text = normalized
            best_score = adjusted_score

    return best_text, best_score


def _select_best_candidate(candidates):
    best_text, _ = _select_best_candidate_detail(candidates)
    return best_text


def _run_ocr(image):
    candidates = _easyocr_candidates(image)
    best_text, best_score = _select_best_candidate_detail(candidates)
    if best_text and best_score >= FAST_OCR_SCORE:
        return best_text

    candidates.extend(_tesseract_candidates(image))
    best_text = _select_best_candidate(candidates)
    if best_text:
        return best_text

    try:
        results = _reader.readtext(image, **OCR_READ_KWARGS)
        text, _ = _compose_text_from_results(results)
        return text
    except Exception:
        return ''


def extract_text(image_path):
    try:
        image = _load_cv_image(image_path)
        return _run_ocr(image)
    except Exception:
        try:
            if pytesseract is not None:
                try:
                    text = pytesseract.image_to_string(image_path, config='--oem 3 --psm 6', lang='eng')
                    cleaned = _clean_ocr_text(text)
                    if cleaned:
                        return cleaned
                except Exception:
                    pass
            results = _reader.readtext(image_path, **OCR_READ_KWARGS)
            text, _ = _compose_text_from_results(results)
            return text
        except Exception:
            return ''


# ---------------------------------------------------------------------------
# Lightweight rule-based text simplifier
# ---------------------------------------------------------------------------
COMPLEX_WORD_MAP = {
    r'\butilize\b': 'use', r'\bfacilitate\b': 'help', r'\bimplement\b': 'do',
    r'\bdemonstrate\b': 'show', r'\bpurchase\b': 'buy', r'\bobtain\b': 'get',
    r'\bcommence\b': 'start', r'\bterminate\b': 'end', r'\bsubsequent\b': 'next',
    r'\bprior to\b': 'before', r'\bin order to\b': 'to',
    r'\bwith regard to\b': 'about', r'\bin spite of\b': 'despite',
    r'\bdue to the fact that\b': 'because', r'\bat this point in time\b': 'now',
    r'\bbiochemical\b': 'chemical', r'\bphotosynthesis\b': 'food-making',
    r'\bhowever\b': 'but', r'\btherefore\b': 'so', r'\bnevertheless\b': 'still',
    r'\bconsequently\b': 'so', r'\bfurthermore\b': 'also', r'\bmoreover\b': 'also',
    r'\bsignifies\b': 'means', r'\bexamining\b': 'looking at',
    r'\bemphasizes\b': 'stresses', r'\bintegration\b': 'use',
    r'\bconversational\b': 'chat-based', r'\bacademic\b': 'school',
    r'\bworkflows\b': 'processes', r'\breferencing\b': 'citing',
    r'\banalyzes\b': 'looks at', r'\bhighlighting\b': 'pointing out',
    r'\baccessibility\b': 'ease of access', r'\bpedagogical\b': 'teaching',
    r'\bstrategies\b': 'plans', r'\bpractices\b': 'methods',
}


def simplify_sentence(sentence):
    result = sentence
    for pattern, replacement in COMPLEX_WORD_MAP.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def split_long_sentence(sentence, max_words=20):
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    split_points = [' and ', ' but ', ' because ', ' which ', ' that ', ' so ']
    for sp in split_points:
        if sp in sentence.lower():
            idx = sentence.lower().find(sp)
            parts = [sentence[:idx].strip(), sentence[idx + len(sp):].strip()]
            if parts[1]:
                parts[1] = parts[1][0].upper() + parts[1][1:]
            return [p for p in parts if p]
    return [sentence]


def simplify_text_logic(text):
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    simplified_sentences = []
    for sent in raw_sentences:
        if not sent.strip():
            continue
        simplified = simplify_sentence(sent)
        parts = split_long_sentence(simplified)
        simplified_sentences.extend(parts)
    return ' '.join(simplified_sentences)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def wants_json_response(request):
    accept = request.headers.get('Accept', '')
    return (
        request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        or 'application/json' in accept
    )


def process_uploaded_image(image_file):
    doc = Document.objects.create(image=image_file)
    text = extract_text(doc.image.path)
    doc.extracted_text = text
    doc.save(update_fields=['extracted_text'])
    return doc, text


def upload_image(request):
    if request.method == 'POST':
        image = request.FILES.get('image')
        if not image:
            if wants_json_response(request):
                return JsonResponse({'error': 'No image provided.'}, status=400)
            return render(request, 'upload.html', {'error_message': 'No image provided.'})

        try:
            doc, text = process_uploaded_image(image)
        except Exception as e:
            error_msg = str(e)
            if wants_json_response(request):
                return JsonResponse({'error': error_msg}, status=500)
            return render(request, 'upload.html', {'error_message': error_msg})

        if wants_json_response(request):
            return JsonResponse({
                'document_id': doc.id,
                'text': text,
                'image_url': doc.image.url,
            })
        return render(request, 'result.html', {'text': text})

    return render(request, 'upload.html')


@require_POST
def simplify_view(request):
    try:
        body = json.loads(request.body)
        original = body.get('text', '').strip()
        if not original:
            return JsonResponse({'error': 'No text provided.'}, status=400)
        simplified = simplify_text_logic(original)
        return JsonResponse({'simplified': simplified})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def history_view(request):
    docs = Document.objects.exclude(extracted_text='').order_by('-id')[:1]
    results = [{'id': d.id, 'text': d.extracted_text[:500]} for d in docs]
    return JsonResponse({'history': results})


@require_POST
def clear_history(request):
    count, _ = Document.objects.all().delete()
    return JsonResponse({'success': True, 'deleted': count})
