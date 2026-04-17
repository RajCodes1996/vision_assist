import json
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .models import Document


def _load_cv_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Could not read the image file.')
    return image


def _deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 240))
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
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = _deskew_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    return thresh


def extract_text(image_path):
    try:
        image = _load_cv_image(image_path)
        processed = _preprocess_for_ocr(image)
        return pytesseract.image_to_string(processed, config='--oem 3 --psm 6')
    except Exception:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img)


# ---------------------------------------------------------------------------
# Lightweight rule-based text simplifier (no heavy ML model needed)
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
    """Apply word-level simplifications to a single sentence."""
    result = sentence
    for pattern, replacement in COMPLEX_WORD_MAP.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def split_long_sentence(sentence, max_words=20):
    """Break a run-on sentence at conjunctions if it exceeds max_words."""
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
    """Full simplification pipeline."""
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

        doc, text = process_uploaded_image(image)

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
    """AJAX endpoint: receives JSON {text: '...'} and returns simplified version."""
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
    """Returns the last scanned document as JSON for voice history command."""
    docs = Document.objects.exclude(extracted_text='').order_by('-id')[:1]
    results = [{'id': d.id, 'text': d.extracted_text[:500]} for d in docs]
    return JsonResponse({'history': results})


@require_POST
def clear_history(request):
    """Deletes all scanned documents from the database."""
    count, _ = Document.objects.all().delete()
    return JsonResponse({'success': True, 'deleted': count})
