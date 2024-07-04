import uuid
from django.shortcuts import render, redirect
from .models import Image
from canvas_ui.forms import ImageUploadForm
from canvas_bgrem.utils import remove_background
from django.core.files.base import ContentFile
from django.utils.timezone import now
from django.urls import reverse
from django.http import HttpResponse
from django.shortcuts import get_object_or_404

from PIL import UnidentifiedImageError
import logging

# Initialize logging
logger = logging.getLogger(__name__)

# Create your views here.

def upload_and_process_image(request):
    context = {'form': ImageUploadForm()}
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            original_name = request.FILES['image'].name
            extension = original_name.split('.')[-1].lower()
            if extension not in ['jpg', 'jpeg', 'png']:
                context['error'] = 'Unsupported image format. Please upload a JPG or PNG file.'
                return render(request, 'canvas_ui/index.html', context)

            unique_filename = f"{uuid.uuid4()}_{now().strftime('%Y%m%d%H%M%S')}.{extension}"
            request.FILES['image'].name = unique_filename

            input_image_instance = Image(input_image=request.FILES['image'])
            input_image_instance.save()

            try:
                output_image_bytes = remove_background(request.FILES['image'])
                processed_filename = f"{unique_filename.split('.')[0]}_processed.png"
                input_image_instance.output_image.save(
                    processed_filename,
                    ContentFile(output_image_bytes)
                )
                input_image_instance.save()

                context['original_image_url'] = input_image_instance.input_image.url
                context['processed_image_url'] = input_image_instance.output_image.url
                context['download_url'] = reverse('download_image', args=[input_image_instance.output_image.name])
            except UnidentifiedImageError as e:
                logger.error(f"Failed to process image: {e}")
                context['error'] = 'Failed to process the image. Please try again with a different file.'
                return render(request, 'canvas_ui/index.html', context)

            return render(request, 'canvas_ui/index.html', context)
    return render(request, 'canvas_ui/index.html', context)

def download_image(request, image_name):
    # Find the image instance by matching the output_image's file name
    image_instance = get_object_or_404(Image, output_image__endswith=image_name)
    
    # Get the path of the output_image
    image_path = image_instance.output_image.path
    
    # Read the image file
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Create an HTTP response with the image data and set the MIME type to image/png
    response = HttpResponse(image_data, content_type='image/png')
    response['Content-Disposition'] = f'attachment; filename="{image_name}"'
    return response