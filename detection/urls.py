from django.urls import path
from .views import HealthPredictionView

urlpatterns = [
    path("hfprediction/", HealthPredictionView.as_view(), name="hfprediction"),
]