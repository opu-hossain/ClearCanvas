from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_and_process_image, name='upload-image'),
    path('download/<path:image_name>', views.download_image, name='download_image')
]