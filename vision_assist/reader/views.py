import json
import re
from statistics import mean

import cv2
import easyocr
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

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
}


def _load_cv_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Could not read the image file.')
    return image


def _deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _resize_for_ocr(image, min_width=1200):
    height, width = image.shape[:2]
    if width >= min_width:
        return image
    scale = min_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def _prepare_fast_variants(image):
    base = _resize_for_ocr(image)
    deskewed = _deskew_image(base)
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11,
    )

    return [
        ('base', gray),
        ('enhanced', enhanced),
        ('adaptive', adaptive),
    ]


def _clean_ocr_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _compose_text_from_results(results):
    if not results:
        return '', 0.0

    items = []
    for bbox, text, confidence in results:
        cleaned = text.strip()
        if not cleaned:
            continue
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        items.append({
            'text': cleaned,
            'x': min(xs),
            'y': min(ys),
            'h': max(ys) - min(ys),
            'confidence': float(confidence or 0.0),
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
            formatted_lines.append(line_text)
            confidences.extend(item['confidence'] for item in ordered)

    text = _clean_ocr_text('\n'.join(formatted_lines))
    if not text:
        return '', 0.0

    alpha_ratio = sum(ch.isalnum() or ch.isspace() for ch in text) / max(len(text), 1)
    score = (mean(confidences) if confidences else 0.0) * 100 + len(text) + alpha_ratio * 20
    return text, score


def _run_ocr(image):
    fallback_text = ''
    for index, (_, variant) in enumerate(_prepare_fast_variants(image)):
        try:
            results = _reader.readtext(variant, **OCR_READ_KWARGS)
            text, score = _compose_text_from_results(results)
            if text:
                if index == 0:
                    return text
                if score >= 28:
                    return text
                fallback_text = text
        except Exception:
            continue

    if fallback_text:
        return fallback_text

    results = _reader.readtext(image, **OCR_READ_KWARGS)
    text, _ = _compose_text_from_results(results)
    return text


def extract_text(image_path):
    try:
        image = _load_cv_image(image_path)
        return _run_ocr(image)
    except Exception:
        try:
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
