🌱 PlantCare --- Plant Disease & Crop Management System

PlantCare is an AI-powered smart farming web application built with
Flask. It combines plant disease detection, crop management,
personalized crop scheduling, email reminders, planting guidance,
multilingual support, voice assistance, and an AI Farmer Assistant in
one integrated platform.

The application is deployed as a web service and uses a cloud PostgreSQL
database. For lightweight production inference, the deployed
disease-detection system uses ONNX Runtime.

✨ Features

AI-based plant disease detection from uploaded or captured leaf
images

38-class plant disease classification

Guest disease detection without login

Confidence-based uncertain / out-of-distribution rejection

User registration, login, logout, and password reset

Crop management and crop history

Personalized crop-care schedules

Watering, compost, fertilizer, inspection, and harvest activities

Crop schedule confirmation and reminder emails

Disease scan history for logged-in users

Planting techniques and AI-generated plant guides

English, Marathi, and Hindi planting guidance

Text-to-speech guide playback

Groq-powered AI Farmer Assistant

User-specific AI chat history

PostgreSQL database integration

Responsive interface for desktop and mobile

🧠 AI Disease Detection Model

The project was originally trained and fine-tuned using
TensorFlow/Keras. For deployment on a resource-limited cloud service,
the selected model is converted to ONNX and loaded using ONNX Runtime.

Model files

model/
├── realworld_best_model.keras
├── realworld_best_model.onnx
├── final_plant_disease_model.keras
└── class_names.json

realworld_best_model.keras is retained as the Keras version of the
selected model, while realworld_best_model.onnx is used for
lightweight production inference.

The model supports 38 crop/disease classes. Class labels are stored in:

model/class_names.json

The prediction pipeline applies a confidence threshold so that highly
uncertain images can be rejected instead of always returning a disease
class.

📊 Model Evaluation

The project includes separate evaluation material so that model
performance can be reviewed beyond training accuracy.

Evaluation/
├── test data.csv
└── MODEL_EVALUATION.md

Evaluation includes dataset-style images, external real-world images,
and uncertain / out-of-distribution images.

The project also experimented with additional real-world fine-tuning
using PlantWild images. This helped improve predictions for some
external samples, while testing showed that real-world disease
recognition remains more challenging than dataset-style classification
because of background, lighting, camera angle, image quality, leaf
orientation, and disease severity.

See:

Evaluation/MODEL_EVALUATION.md

for the recorded evaluation results.

🛠️ Tech Stack

Backend

Python

Flask

Flask-SQLAlchemy

PostgreSQL

Flask-Bcrypt

APScheduler

Gunicorn

AI / ML

TensorFlow / Keras --- model development and training

ONNX Runtime --- deployed model inference

NumPy

Pillow

AI Services

Groq API --- AI Farmer Assistant and AI-supported guidance

gTTS --- planting-guide voice generation

Email

Brevo Transactional Email API

HTTP-based email delivery for cloud deployment

Frontend

HTML

CSS

JavaScript

Jinja2

Font Awesome

Deployment

GitHub --- source-code repository

Render --- Flask web-service deployment

Render PostgreSQL --- production database

📁 Project Structure

Plant-Disease-Crop-Management/
│
├── app.py
├── plant_ai.py
├── farmer_chat.py
├── crop_rules.py
├── disease_info.py
├── planting_data.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── Evaluation/
│   ├── test data.csv
│   └── MODEL_EVALUATION.md
│
├── model/
│   ├── final_plant_disease_model.keras
│   ├── realworld_best_model.keras
│   ├── realworld_best_model.onnx
│   └── class_names.json
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── forgot_password.html
│   ├── reset_password.html
│   ├── dashboard.html
│   ├── detect.html
│   ├── result.html
│   ├── uncertain_result.html
│   ├── add_crop.html
│   ├── manage_crops.html
│   ├── crop_history.html
│   ├── reminders.html
│   ├── disease_history.html
│   ├── planting_techniques.html
│   ├── schedule_result.html
│   └── profile.html
│
└── static/
    ├── css/
    │   └── style.css
    ├── images/
    │   └── farmer.jpg
    └── js/

⚙️ Environment Variables

Create a .env file in the project root for local development.

SECRET_KEY=replace-with-a-strong-secret-key

DB_HOST=localhost
DB_PORT=5432
DB_NAME=plantcare
DB_USER=postgres
DB_PASSWORD=your_database_password

GROQ_API_KEY=your_groq_api_key

BREVO_API_KEY=your_brevo_api_key
BREVO_SENDER_EMAIL=your_verified_sender_email
BREVO_SENDER_NAME=PlantCare

Do not upload .env to GitHub.

For production, configure these values directly in the hosting
platform's environment-variable settings.

🚀 Local Installation

1. Clone the repository

git clone <your-repository-url>
cd Plant-Disease-Crop-Management

2. Create a virtual environment

Windows PowerShell:

python -m venv venv
.\venv\Scripts\Activate.ps1

Linux/macOS:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Configure PostgreSQL

Create a PostgreSQL database and add its credentials to .env.

The application uses SQLAlchemy for database access and creates required
tables during application initialization.

5. Run locally

python app.py

Open:

http://127.0.0.1:5000

📧 Email System

PlantCare uses the Brevo Transactional Email API for deployed email
delivery.

Email features include:

Password reset emails

Crop schedule confirmation emails

Crop-care reminder emails

Required environment variables:

BREVO_API_KEY=your_brevo_api_key
BREVO_SENDER_EMAIL=your_verified_sender_email
BREVO_SENDER_NAME=PlantCare

The sender email must be verified in Brevo.

Using an HTTP email API avoids depending on SMTP connectivity in the
deployed web service.

🤖 AI Farmer Assistant

The AI Farmer Assistant uses the Groq API for farming-related questions.

It supports:

English

Marathi

Hindi

Configure:

GROQ_API_KEY=your_groq_api_key

Chat history is stored per logged-in user.

🌿 Planting Guide & Voice Assistance

PlantCare provides planting guidance covering crop-care steps and
farming information.

The guide supports English, Marathi, and Hindi. Voice playback is
generated using gTTS, so internet connectivity is required when audio
needs to be generated.

🌾 Crop Management & Scheduler

Logged-in users can:

Add and manage crops

Create personalized crop schedules

Track watering dates

Track compost dates

Track fertilizer dates

Track inspection dates

View expected harvest dates

View crop history

Mark crop-care activities as completed

Track pending and overdue activities

Enable email notifications

Disease detection remains accessible to guest users without requiring an
account.

☁️ Production Deployment

The project is deployed using Render.

Build command

pip install -r requirements.txt

Start command

gunicorn app:app --timeout 180

Production requirements

Use cloud PostgreSQL instead of localhost

Store secrets only in environment variables

Keep Flask debug mode disabled

Use Gunicorn instead of the Flask development server

Include the ONNX model and class_names.json

Configure the Groq API key

Configure the Brevo API key and verified sender

Keep .env out of Git

Why ONNX Runtime?

The original TensorFlow/Keras runtime requires considerably more
deployment resources. The production version therefore uses the
converted ONNX model with ONNX Runtime to provide lighter CPU-based
inference while retaining the trained model for the disease-detection
workflow.

🔐 Security Notes

Never commit:

Database passwords

SECRET_KEY

Groq API keys

Brevo API keys

Email credentials

Any other private production credentials

Recommended .gitignore:

venv/
.env
__pycache__/
*.pyc
.vscode/
.DS_Store

The model/ directory should not be ignored when the deployment depends
on model files stored in the repository.

📌 Main Modules

Disease Detection --- analyzes leaf images and predicts supported
plant diseases.

Crop Management --- stores and manages crops associated with
individual users.

Crop Scheduler --- generates personalized crop-care schedules.

Reminders --- tracks pending, due, and overdue crop-care activities.

Planting Guide --- provides crop-specific growing and care guidance.

Voice Assistance --- converts supported planting guidance into
speech.

AI Farmer Assistant --- answers farming-related questions using
Groq.

Disease History --- stores disease scans for authenticated users.

Authentication --- supports registration, login, logout, and
password-reset workflows.

⚠️ Limitations

Real-world images can be more difficult to classify than controlled
dataset images.

Prediction quality can be affected by lighting, background, blur,
camera angle, leaf visibility, and disease severity.

Only diseases represented by the supported model classes can be
classified.

AI-generated farming guidance should be treated as decision support
rather than professional diagnosis.

Third-party features such as Groq, Brevo, and gTTS require internet
access and depend on their respective service availability.

🔮 Future Scope

Possible future improvements include:

Larger real-world disease datasets

Additional crops and disease classes

Improved out-of-distribution detection

Weather-aware crop recommendations

Location-aware farming guidance

Mobile application support

Advanced notification scheduling

Model optimization and continuous evaluation with diverse field
images

⚠️ Disclaimer

PlantCare is an educational and decision-support project. Disease
predictions, treatment suggestions, and AI-generated farming guidance
should not replace professional agricultural advice, laboratory
diagnosis, or official pesticide-label instructions.

👩‍💻 Developed By

Nutan Salunkhe

PlantCare --- AI-powered plant disease detection and smart crop
management.