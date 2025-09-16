from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import tempfile
import os

# Import your pipeline
from codes.pipeline import Pipeline
import joblib


# Paths to your trained models and scaler
CNN_MODEL_PATH = os.path.join("codes", "cnn_model.h5")
TRADITIONAL_ML_MODEL_PATH = os.path.join("codes", "ml_model.joblib")
SCALAR_PATH = os.path.join("codes", "scaler.pk1")
ecg_path = os.path.join("codes", "01001_hr.dat")

# Load scaler
scalar = joblib.load(SCALAR_PATH)

# Initialize pipeline
pipeline = Pipeline(CNN_MODEL_PATH, TRADITIONAL_ML_MODEL_PATH, scalar)

def save_temp_file(file_obj, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in file_obj.chunks():
        tmp.write(chunk)
    tmp.close()
    return tmp.name

def save_ecg_files(dat_file, hea_file):
        # Create temp folder
    temp_dir = tempfile.mkdtemp()

    # Use the **original base name** of the uploaded files
    dat_base_name = os.path.splitext(dat_file.name)[0]
    dat_path = os.path.join(temp_dir, dat_base_name + ".dat")
    hea_path = os.path.join(temp_dir, dat_base_name + ".hea")

    # Save .dat
    with open(dat_path, "wb") as f:
        for chunk in dat_file.chunks():
            f.write(chunk)

    # Save .hea
    with open(hea_path, "wb") as f:
        for chunk in hea_file.chunks():
            f.write(chunk)

    # Return path without extension for WFDB
    return os.path.join(temp_dir, dat_base_name)

class HealthPredictionView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        age = request.data.get("age")
        sex = request.data.get("sex")
        dat_file = request.FILES.get("ecg_file")
        hea_file = request.FILES.get("ecg_header")

        if not (age and sex and dat_file and hea_file):
            return Response({"error": "Missing input(s)"}, status=400)

        try:
            record_path = save_ecg_files(dat_file, hea_file)
            # Run prediction
            probability = pipeline.predict(record_path, int(age), float(sex))  # sex as numeric if needed

            return Response({
                "probability": probability[0][1]           
                })

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
