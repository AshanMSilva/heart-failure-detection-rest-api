# Heart Failure Detection REST API

This project provides a Django REST API for predicting **heart failure** risk using a hybrid deep learning + traditional ML pipeline.  
The API accepts **12-lead ECG signals** (in `.dat` + `.hea` format), along with **age** and **sex**, and returns a probability prediction.

---

## 🚀 Features

- Django REST Framework API with a single endpoint:
  - `POST /api/hfprediction/`
- Inputs:
  - `age` (integer)
  - `sex` (0 = female, 1 = male)
  - `ecg_file` → ECG signal file (`.dat`)
  - `ecg_header` → ECG header file (`.hea`)
- Output: JSON response with predicted probability of heart failure.

---

## 🛠️ Requirements

- Python 3.9+
- Virtual environment (`venv`)
- Dependencies in `requirements.txt`

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/heart-failure-detection-rest-api.git
cd heart-failure-detection-rest-api
```

### 2. Create and Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Make sure pip is upgraded:

```bash
pip install --upgrade pip
```

Install required packages:

```bash
pip install -r requirements.txt
```

⚠️ Important: TensorFlow 2.13 is required for model compatibility

### 4. Start Django Server

```bash
python manage.py runserver
```

The API will be available at:
👉 http://127.0.0.1:8000/api/healthprediction/

## 🧪 Testing with Postman

Open Postman.

Create a POST request to:

http://127.0.0.1:8000/api/healthprediction/

Go to Body → form-data and add:

- Key: age → Type: Text → Value: 45

- Key: sex → Type: Text → Value: 1

- Key: ecg_file → Type: File → Upload your .dat ECG file

- Key: ecg_header → Type: File → Upload the corresponding .hea file

⚠️ Both ecg_file and ecg_header must belong to the same record (e.g., 21001_hr.dat and 21001_hr.hea).

Click Send.

Example Response:

```bash
{
  "prediction": 0.82
}
```

## 📂 Project Structure

heart-failure-detection-rest-api/
│
├── codes/ # Saved models (CNN, ML, scaler)
├── detection/ # Django app with views & urls
├── hf_fdetection_api/ # Django project settings
├── requirements.txt # Python dependencies
├── manage.py
└── README.md

## 🧑‍💻 Development Notes

- Both .dat and .hea must be uploaded for WFDB to read the ECG signal.

- Make sure your .h5 model is re-saved with TensorFlow 2.13 for compatibility.

- The pipeline handles preprocessing (filtering, normalization) internally.
