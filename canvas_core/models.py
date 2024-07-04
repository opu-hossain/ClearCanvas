from django.db import models

# Create your models here.

class Image(models.Model):
    input_image = models.ImageField(upload_to='inputs/')
    output_image = models.ImageField(upload_to='outputs/', null=True, blank=True)