from types import SimpleNamespace
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from .models import Document
from .views import (
    _compose_text_from_results,
    _normalize_technical_phrase_order,
    _select_best_candidate,
)


class ReaderViewTests(TestCase):
    def test_upload_page_prompts_voice_first_navigation(self):
        response = self.client.get(reverse('upload'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Say Start Camera')
        self.assertContains(response, 'Read Last Scan')

    def test_upload_json_without_file_returns_error(self):
        response = self.client.post(
            reverse('upload'),
            HTTP_ACCEPT='application/json',
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )
        self.assertEqual(response.status_code, 400)
        self.assertJSONEqual(response.content, {'error': 'No image provided.'})

    @patch('reader.views.process_uploaded_image')
    def test_upload_json_returns_processed_payload(self, mock_process):
        mock_process.return_value = (
            SimpleNamespace(
                id=7,
                image=SimpleNamespace(url='/media/images/example.jpg'),
            ),
            'Sample OCR text',
        )

        image = SimpleUploadedFile('scan.jpg', b'image-bytes', content_type='image/jpeg')
        response = self.client.post(
            reverse('upload'),
            {'image': image},
            HTTP_ACCEPT='application/json',
            HTTP_X_REQUESTED_WITH='XMLHttpRequest',
        )

        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(
            response.content,
            {
                'document_id': 7,
                'text': 'Sample OCR text',
                'image_url': '/media/images/example.jpg',
            },
        )

    def test_simplify_view_returns_simplified_text(self):
        response = self.client.post(
            reverse('simplify'),
            data='{"text": "We utilize complex workflows."}',
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(response.content, {'simplified': 'We use complex processes.'})

    def test_history_and_clear_history_routes_work(self):
        Document.objects.create(image='images/test-1.jpg', extracted_text='First scan')
        Document.objects.create(image='images/test-2.jpg', extracted_text='Second scan')

        history_response = self.client.get(reverse('history'))
        self.assertEqual(history_response.status_code, 200)
        self.assertJSONEqual(
            history_response.content,
            {'history': [{'id': 2, 'text': 'Second scan'}]},
        )

        clear_response = self.client.post(reverse('clear_history'))
        self.assertEqual(clear_response.status_code, 200)
        self.assertJSONEqual(clear_response.content, {'success': True, 'deleted': 2})
        self.assertEqual(Document.objects.count(), 0)

    def test_compose_text_removes_stray_digit_from_short_label(self):
        results = [
            (
                [[0, 0], [40, 0], [40, 20], [0, 20]],
                'KAKU',
                0.97,
            ),
            (
                [[48, 0], [58, 0], [58, 20], [48, 20]],
                '5',
                0.41,
            ),
        ]

        text, score = _compose_text_from_results(results)

        self.assertEqual(text, 'KAKU')
        self.assertGreater(score, 0)

    def test_best_candidate_prefers_cleaner_label_over_noisy_variant(self):
        candidates = [
            ('KAKU 5', 91.0),
            ('KAKU', 72.0),
            ('KAKU 3', 75.0),
        ]

        self.assertEqual(_select_best_candidate(candidates), 'KAKU')

    def test_technical_phrase_order_prefers_machine_learning_style_terms(self):
        self.assertEqual(
            _normalize_technical_phrase_order('learning machine'),
            'machine learning',
        )
        self.assertEqual(_normalize_technical_phrase_order('data science'), 'data science')
