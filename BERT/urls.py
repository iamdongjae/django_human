from django.urls import path
from .views import *

app_name = 'BERT'

urlpatterns = [
    path('', index, name="index"),
    path('emotion', emotion, name="emotion"),
    path('emotion_service/<str:text>/', emotion_service, name="emotion_service"),
]
