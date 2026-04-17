# Vision Assist - AI Reader for Blind Users (M.Tech Research Project)

## 📌 Project Overview

Vision Assist is a simple AI-based application designed to help visually
impaired users read text from images using OCR and Text-to-Speech.

------------------------------------------------------------------------

## 🎯 Objectives

-   Extract text from images
-   Convert extracted text to speech
-   Provide a simple accessible interface

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Python
-   Django
-   Tesseract OCR
-   pyttsx3 (Text-to-Speech)
-   HTML, CSS, JavaScript
-   SQLite

------------------------------------------------------------------------

## 🚀 Step-by-Step Development

### Step 1: Environment Setup

``` bash
pip install django pytesseract pillow pyttsx3 SpeechRecognition
```

Install Tesseract OCR separately and set path.

------------------------------------------------------------------------

### Step 2: Create Django Project

``` bash
django-admin startproject vision_assist
cd vision_assist
python manage.py startapp reader
```

------------------------------------------------------------------------

### Step 3: Configure Settings

Add 'reader' to INSTALLED_APPS.

Configure media files:

``` python
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

------------------------------------------------------------------------

### Step 4: Create Model

``` python
class Document(models.Model):
    image = models.ImageField(upload_to='images/')
    extracted_text = models.TextField(blank=True)
```

Run migrations:

``` bash
python manage.py makemigrations
python manage.py migrate
```

------------------------------------------------------------------------

### Step 5: OCR Function

``` python
import pytesseract
from PIL import Image

def extract_text(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)
```

------------------------------------------------------------------------

### Step 6: Text-to-Speech

``` python
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
```

------------------------------------------------------------------------

### Step 7: Create View

``` python
def upload_image(request):
    if request.method == 'POST':
        image = request.FILES['image']
        doc = Document.objects.create(image=image)

        text = extract_text(doc.image.path)
        doc.extracted_text = text
        doc.save()

        speak(text)

        return render(request, 'result.html', {'text': text})

    return render(request, 'upload.html')
```

------------------------------------------------------------------------

### Step 8: Templates

#### upload.html

``` html
<form method="POST" enctype="multipart/form-data">
    {% csrf_token %}
    <input type="file" name="image">
    <button type="submit">Upload & Read</button>
</form>
```

#### result.html

``` html
<h2>Extracted Text:</h2>
<p>{{ text }}</p>
```

------------------------------------------------------------------------

### Step 9: URL Configuration

``` python
path('', views.upload_image, name='upload')
```

------------------------------------------------------------------------

### Step 10: Run Server

``` bash
python manage.py runserver
```

------------------------------------------------------------------------

## 🎤 Optional: Voice Command

Use SpeechRecognition for basic command input (single trigger only).

------------------------------------------------------------------------

## 🧪 Research Enhancements

-   Compare OCR accuracy on different image types
-   Measure processing time
-   Test multiple languages

------------------------------------------------------------------------

## 📊 Evaluation Metrics

-   Accuracy (%)
-   Response Time
-   User Experience

------------------------------------------------------------------------

## ⚠️ Limitations

-   Requires clear images
-   Limited voice command support
-   Offline TTS limitations

------------------------------------------------------------------------

How Your Project Works (Short & Clear)
🧑‍🦯 Step 1: User gives input
User uploads an image (or later: uses camera / voice command)
📸 Step 2: Image goes to backend
Django receives the image
Stores it in database/media folder
🔍 Step 3: Text Extraction (OCR)
Your system uses Tesseract
It reads the image and extracts text

👉 Example:
Image → "Welcome to School"

🔊 Step 4: Text to Speech
Extracted text is sent to pyttsx3
System converts text → voice

👉 Output:
🎧 “Welcome to School” (spoken aloud)

🖥️ Step 5: Show result (optional)
Text is also displayed on screen
🧠 In One Line

👉 Image → Text (OCR) → Voice (TTS)

🎯 Real-Life Flow
Blind user clicks “Upload” (or says command)
App reads image
App speaks the content
User listens 🎧
------------------------------------------------------------------------

## ✅ Conclusion

This project demonstrates a practical AI application for accessibility
using simple ML techniques and web technologies.

------------------------------------------------------------------------

**End of Document**
