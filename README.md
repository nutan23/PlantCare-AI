🌱 PlantCare — Plant Disease & Crop Management System

PlantCare is an AI-powered smart farming web application built with Flask. It combines plant disease detection, crop scheduling, email reminders, planting guidance, multilingual support, and an AI Farmer Assistant in one platform.

✨ Features

AI-based plant disease detection from uploaded or captured leaf images

38-class TensorFlow/Keras disease classification model

Guest disease detection without login

User registration, login, logout, and password reset by email

Crop management and crop history

Personalized crop schedules

Watering, compost, fertilizer, inspection, and harvest activities

Daily email crop-care reminders

Disease scan history

Planting techniques and AI-generated plant guides

English, Marathi, and Hindi planting guidance

Text-to-speech guide playback

Groq-powered AI Farmer Assistant

User-specific AI chat history

PostgreSQL database integration

Responsive interface for desktop and mobile


🧠 AI Model

The disease prediction system uses a TensorFlow/Keras model stored in:

model/realworld_best_model.keras

Class names are stored in:

model/class_names.json

The model predicts plant crop/disease classes and applies a confidence threshold so uncertain predictions can be rejected instead of always returning a disease class.

🛠️ Tech Stack

Backend

Python

Flask

Flask-SQLAlchemy

PostgreSQL

Flask-Bcrypt

Flask-Mail

APScheduler

AI / ML

TensorFlow / Keras

NumPy

Pillow

AI Services

Groq API

gTTS

Frontend

HTML

CSS

JavaScript

Jinja2

Font Awesome

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
├── .env
│
├── model/
│   ├── realworld_best_model.keras
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
│   └── profile.html
│
└── static/
    ├── css/
    │   └── style.css
    ├── images/
    └── js/

⚙️ Environment Variables

Create a .env file in the project root.

SECRET_KEY=replace-with-a-strong-secret-key

DB_HOST=localhost
DB_PORT=5432
DB_NAME=plantcare
DB_USER=postgres
DB_PASSWORD=your_database_password

MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_gmail_app_password

GROQ_API_KEY=your_groq_api_key

Do not upload the .env file to GitHub.

🚀 Local Installation

1. Clone the repository

git clone <your-repository-url>
cd Plant-Disease-Crop-Management

2. Create a virtual environment

Windows:

python -m venv venv
.\venv\Scripts\Activate.ps1

Linux/macOS:

python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Configure PostgreSQL

Create a PostgreSQL database and add its credentials to .env.

The application uses SQLAlchemy and creates missing tables when the app starts.

5. Run the application

python app.py

Open:

http://127.0.0.1:5000

📧 Email Reminders

PlantCare can send:

password reset emails

crop schedule confirmation emails

daily crop-care reminder emails

For Gmail, use a Google App Password instead of your normal Gmail password.

🤖 AI Farmer Assistant

The AI Farmer Assistant uses the Groq API and supports farming-related questions in:

English

Marathi

Hindi

Set GROQ_API_KEY in .env before running the chatbot.


🌿 Planting Guide Voice

Planting guide audio is generated using gTTS.

An internet connection is required for gTTS voice generation.


☁️ Deployment Notes

For production deployment:

Keep .env out of Git.

Add environment variables in the deployment platform dashboard.

Use PostgreSQL hosted in the cloud instead of localhost.

Use Gunicorn instead of the Flask development server.

Set Flask debug mode to False in production.

Make sure the .keras model file is included and fits within the hosting platform's storage/build limits.

Keep model/class_names.json deployed beside the model.

Configure Gmail/Groq credentials as environment variables.

Recommended production start command:

gunicorn app:app --timeout 180

The longer timeout helps because TensorFlow model loading can take more time during application startup.


🔐 Security Notes

Before production deployment:

Move SECRET_KEY to an environment variable.

Never commit database passwords, Gmail App Passwords, or Groq API keys.

Keep .env in .gitignore.

Disable Flask debug mode.


Suggested .gitignore:

venv/
.env
__pycache__/
*.pyc
.DS_Store

Do not ignore the model directory if the model must be deployed with the application.

📌 Main Modules

Disease Detection — predicts plant disease from leaf images

Crop Management — stores and manages user crops

Crop Scheduler — generates crop-care schedules

Reminders — tracks pending, overdue, and upcoming activities

Planting Guide — provides step-by-step growing guidance

AI Farmer Assistant — answers farming-related questions

Disease History — stores logged-in users' disease scans

⚠️ Disclaimer

PlantCare is an educational and decision-support project. Disease predictions and treatment guidance should not replace professional agricultural advice, laboratory diagnosis, or official pesticide-label instructions.

Built as an AI-powered smart farming and crop-management project.

Developed By
Nutan Salunkhe
