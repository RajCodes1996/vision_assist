from django.db import models

class Document(models.Model):
    image = models.ImageField(upload_to='images/')
    extracted_text = models.TextField(blank=True)
