from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
    send_file
)

from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from dotenv import load_dotenv
from sqlalchemy import URL
from itsdangerous import (
    URLSafeTimedSerializer,
    SignatureExpired,
    BadSignature
)
from datetime import datetime, timedelta
from crop_rules import get_crop_rule
from plant_ai import generate_planting_guide
from farmer_chat import get_farmer_ai_response
import os

import json
import base64
import io
import numpy as np
import onnxruntime as ort
from disease_info import DISEASE_INFO, DEFAULT_INFO

from PIL import Image
from werkzeug.utils import secure_filename
from apscheduler.schedulers.background import BackgroundScheduler
from gtts import gTTS

import tempfile

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# SECRET KEY & SESSION CONFIGURATION
# ==========================================

app.config["SECRET_KEY"] = "plantcare-secret-key"

# Session is NOT permanent.
# New browser session will require login again.
app.config["SESSION_PERMANENT"] = False


# ==========================================
# POSTGRESQL DATABASE CONFIGURATION
# ==========================================

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


database_url = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME
)


app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


# ==========================================
# MAIL CONFIGURATION
# ==========================================

app.config["MAIL_SERVER"] = os.getenv(
    "MAIL_SERVER",
    "smtp.gmail.com"
)

app.config["MAIL_PORT"] = int(
    os.getenv("MAIL_PORT", 587)
)

app.config["MAIL_USE_TLS"] = (
    os.getenv("MAIL_USE_TLS", "True").lower() == "true"
)

app.config["MAIL_USERNAME"] = os.getenv(
    "MAIL_USERNAME"
)

app.config["MAIL_PASSWORD"] = os.getenv(
    "MAIL_PASSWORD"
)


# ==========================================
# INITIALIZE EXTENSIONS
# ==========================================

db = SQLAlchemy(app)

bcrypt = Bcrypt(app)

mail = Mail(app)

# ==========================================
# PLANT DISEASE AI MODEL
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "realworld_best_model.onnx"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.json"
)

print("🌿 Loading Plant Disease ONNX Model...")

plant_model = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

model_input_name = (
    plant_model
    .get_inputs()[0]
    .name
)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)

print(
    f"✅ Plant Disease Model Loaded - "
    f"{len(class_names)} classes"
)

# ==========================================
# PASSWORD RESET TOKEN
# ==========================================

serializer = URLSafeTimedSerializer(
    app.config["SECRET_KEY"]
)


# ==========================================
# USER MODEL
# ==========================================

class User(db.Model):

    __tablename__ = "users"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    mobile = db.Column(
        db.String(15),
        unique=True,
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    preferred_language = db.Column(
        db.String(20),
        default="English"
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

# ==========================================
# CROP MODEL
# ==========================================

class Crop(db.Model):

    __tablename__ = "crops"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    crop_name = db.Column(
        db.String(100),
        nullable=False
    )

    variety = db.Column(
        db.String(100),
        nullable=True
    )

    planting_date = db.Column(
        db.Date,
        nullable=False
    )

    farm_name = db.Column(
        db.String(120),
        nullable=True
    )

    area = db.Column(
        db.String(50),
        nullable=True
    )

    soil_type = db.Column(
        db.String(50),
        nullable=True
    )

    irrigation_type = db.Column(
        db.String(50),
        nullable=True
    )

    notes = db.Column(
        db.Text,
        nullable=True
    )

    status = db.Column(
        db.String(30),
        default="Active"
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

# ==========================================
# crop shedule model
# ==========================================
class CropSchedule(db.Model):

    __tablename__ = "crop_schedules"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    crop_name = db.Column(
        db.String(100),
        nullable=False
    )

    variety = db.Column(
        db.String(100)
    )

    planting_date = db.Column(
        db.Date,
        nullable=False
    )

    farm_name = db.Column(
        db.String(120)
    )

    area = db.Column(
        db.String(50)
    )

    soil_type = db.Column(
        db.String(50)
    )

    irrigation_type = db.Column(
        db.String(50)
    )

    crop_stage = db.Column(
        db.String(50)
    )

    last_watering_date = db.Column(
        db.Date
    )

    last_compost_date = db.Column(
        db.Date
    )

    last_fertilizer_date = db.Column(
        db.Date
    )

    last_inspection_date = db.Column(
        db.Date
    )

    next_watering_date = db.Column(
        db.Date
    )

    next_compost_date = db.Column(
        db.Date
    )

    next_fertilizer_date = db.Column(
        db.Date
    )

    next_inspection_date = db.Column(
        db.Date
    )

    expected_harvest_date = db.Column(
        db.Date
    )

    notify_me = db.Column(
        db.Boolean,
        default=False
    )

    status = db.Column(
        db.String(30),
        default="Active"
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    # ==========================================
# CROP ACTIVITY MODEL
# ==========================================

class CropActivity(db.Model):

    __tablename__ = "crop_activities"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    schedule_id = db.Column(
        db.Integer,
        db.ForeignKey("crop_schedules.id"),
        nullable=False
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    activity_type = db.Column(
        db.String(50),
        nullable=False
    )

    due_date = db.Column(
        db.Date,
        nullable=False
    )

    completed_date = db.Column(
        db.Date,
        nullable=True
    )

    status = db.Column(
        db.String(30),
        default="Pending"
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

# ==========================================
# DISEASE HISTORY MODEL
# ==========================================

class DiseaseHistory(db.Model):

    __tablename__ = "disease_history"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    crop_name = db.Column(
        db.String(100),
        nullable=False
    )

    disease_name = db.Column(
        db.String(150),
        nullable=False
    )

    confidence = db.Column(
        db.Float,
        nullable=False
    )

    status = db.Column(
        db.String(50),
        nullable=False
    )

    prediction_class = db.Column(
        db.String(200),
        nullable=True
    )

    scan_source = db.Column(
        db.String(30),
        nullable=True
    )

    scanned_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    # ==========================================
# AI CHAT HISTORY MODEL
# ==========================================

class ChatHistory(db.Model):

    __tablename__ = "chat_history"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    question = db.Column(
        db.Text,
        nullable=False
    )

    answer = db.Column(
        db.Text,
        nullable=False
    )

    language = db.Column(
        db.String(10),
        default="en",
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

# ==========================================
# DATE PARSER HELPER
# ==========================================

def parse_optional_date(value):

    if not value:
        return None

    return datetime.strptime(
        value,
        "%Y-%m-%d"
    ).date()


# ==========================================
# CREATE INITIAL CROP ACTIVITIES
# ==========================================

def create_initial_crop_activities(schedule):

    activities = [

        (
            "Watering",
            schedule.next_watering_date
        ),

        (
            "Compost",
            schedule.next_compost_date
        ),

        (
            "Fertilizer",
            schedule.next_fertilizer_date
        ),

        (
            "Inspection",
            schedule.next_inspection_date
        ),

        (
            "Harvest",
            schedule.expected_harvest_date
        )
    ]

    for activity_type, due_date in activities:

        if not due_date:
            continue

        activity = CropActivity(

            schedule_id=schedule.id,

            user_id=schedule.user_id,

            activity_type=activity_type,

            due_date=due_date,

            status="Pending"
        )

        db.session.add(
            activity
        )

    db.session.commit()


# ==========================================
# ENSURE OLD SCHEDULE HAS ACTIVITIES
# ==========================================

def ensure_schedule_activities(schedule):

    existing_count = CropActivity.query.filter_by(
        schedule_id=schedule.id
    ).count()

    if existing_count == 0:

        create_initial_crop_activities(
            schedule
        )


# ==========================================
# SCHEDULE CONFIRMATION EMAIL
# ==========================================

def send_schedule_confirmation_email(
    user,
    schedule
):

    if not user or not user.email:
        return

    message = Message(

        subject=(
            "Welcome to AgreTech - "
            "Crop Schedule Activated"
        ),

        sender=app.config[
            "MAIL_USERNAME"
        ],

        recipients=[
            user.email
        ]
    )

    message.body = f"""
Hello {user.name},

Welcome to AgreTech 🌱

Your crop schedule has been created successfully.

Crop: {schedule.crop_name}
Variety: {schedule.variety or 'Not specified'}
Planting Date: {schedule.planting_date}
Farm / Plot: {schedule.farm_name or 'Not specified'}

Your upcoming crop-care schedule:

💧 Next Watering:
{schedule.next_watering_date}

🌿 Next Compost:
{schedule.next_compost_date}

🧪 Next Fertilizer:
{schedule.next_fertilizer_date}

🔍 Next Inspection:
{schedule.next_inspection_date}

🌾 Expected Harvest:
{schedule.expected_harvest_date}

Email reminders are ENABLED.

Important:
A task will remain pending or overdue
until you mark it as completed in PlantCare.

After you mark an activity as completed,
PlantCare will automatically calculate
its next due date.

Regards,
AgreTech Team 🌿
"""

    mail.send(
        message
    )


# ==========================================
# DAILY CROP REMINDERS
# ==========================================

def send_due_crop_reminders():

    today = datetime.now().date()

    schedules = CropSchedule.query.filter_by(
        status="Active",
        notify_me=True
    ).all()


    for schedule in schedules:

        user = db.session.get(
            User,
            schedule.user_id
        )

        if not user or not user.email:
            continue


        # Make sure older schedules also
        # get activity records
        ensure_schedule_activities(
            schedule
        )


        pending_activities = (

            CropActivity.query

            .filter(
                CropActivity.schedule_id
                == schedule.id,

                CropActivity.user_id
                == schedule.user_id,

                CropActivity.status
                == "Pending",

                CropActivity.due_date
                <= today
            )

            .order_by(
                CropActivity.due_date.asc()
            )

            .all()
        )


        if not pending_activities:
            continue


        task_lines = []


        for activity in pending_activities:

            if activity.due_date < today:

                task_lines.append(
                    f"⚠ OVERDUE - "
                    f"{activity.activity_type} "
                    f"(Due: {activity.due_date})"
                )

            else:

                task_lines.append(
                    f"✅ DUE TODAY - "
                    f"{activity.activity_type} "
                    f"(Due: {activity.due_date})"
                )


        tasks_text = "\n".join(
            task_lines
        )


        message = Message(

            subject=(
                f"AgreTech Reminder - "
                f"{schedule.crop_name}"
            ),

            sender=app.config[
                "MAIL_USERNAME"
            ],

            recipients=[
                user.email
            ]
        )


        message.body = f"""
Hello {user.name},

AgreTech Daily Crop Reminder 🌱

Crop:
{schedule.crop_name}

Farm / Plot:
{schedule.farm_name or 'Not specified'}

The following crop-care activities
need your attention:

{tasks_text}

Please complete the activity and then
open PlantCare → Crop Management →
View History → Mark as Done.

The next activity date will be calculated
from the actual completion date.

Regards,
AgreTech Team 🌿
"""


        try:

            mail.send(
                message
            )

            print(
                "✅ Crop reminder sent:",
                user.email,
                schedule.crop_name
            )


        except Exception as error:

            print(
                "REMINDER MAIL ERROR:",
                error
            )


# ==========================================
# APSCHEDULER JOB
# ==========================================

def scheduled_reminder_job():

    with app.app_context():

        send_due_crop_reminders()

# ==========================================
# AI PLANT GUIDE CACHE MODEL
# ==========================================

class PlantGuide(db.Model):

    __tablename__ = "plant_guides"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    plant_key = db.Column(
        db.String(150),
        unique=True,
        nullable=False,
        index=True
    )

    plant_name = db.Column(
        db.String(150),
        nullable=False
    )

    category = db.Column(
        db.String(80),
        nullable=True
    )

    guide_json = db.Column(
        db.Text,
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ==========================================
# PUBLIC DISEASE DETECTION PAGE
# ==========================================

@app.route("/detect")
def detect():

    return render_template(
        "detect.html"
    )


# ==========================================
# DISEASE PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ======================================
        # GET IMAGE
        # ======================================

        leaf_image = request.files.get(
            "leaf_image"
        )

        captured_image = request.form.get(
            "captured_image"
        )

        image = None

        scan_source = "Unknown"


        # ======================================
        # OPTION 1: UPLOADED IMAGE
        # ======================================

        if (
            leaf_image
            and leaf_image.filename != ""
        ):

            image = Image.open(
                leaf_image.stream
            ).convert("RGB")

            scan_source = "Upload"


        # ======================================
        # OPTION 2: CAMERA IMAGE
        # ======================================

        elif (
            captured_image
            and captured_image.strip() != ""
        ):

            if "," in captured_image:

                captured_image = (
                    captured_image.split(
                        ",",
                        1
                    )[1]
                )

            image_bytes = base64.b64decode(
                captured_image
            )

            image = Image.open(
                io.BytesIO(
                    image_bytes
                )
            ).convert("RGB")

            scan_source = "Camera"


        # ======================================
        # NO IMAGE
        # ======================================

        if image is None:

            flash(
                "Please upload or capture a leaf image.",
                "error"
            )

            return redirect(
                url_for(
                    "detect"
                )
            )


        # ======================================
        # RESIZE IMAGE
        # ======================================

        image = image.resize(
            (224, 224)
        )


        # ======================================
        # IMAGE TO ARRAY
        # ======================================

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ======================================
        # MODEL PREDICTION
        # ======================================

        predictions = plant_model.run(
            None,
            {
                 model_input_name: image_array
            }
        )[0]


        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )


        confidence = float(
            np.max(
                predictions[0]
            )
        ) * 100


        predicted_class = (
            class_names[
                predicted_index
            ]
        )


        # ======================================
        # GET DISEASE INFORMATION
        # ======================================

        disease_info = DISEASE_INFO.get(
            predicted_class,
            DEFAULT_INFO
        )


        # ======================================
        # TOP 3 PREDICTIONS
        # ======================================

        top_indices = np.argsort(
            predictions[0]
        )[-3:][::-1]


        top_predictions = []


        print(
            "\n=============================="
        )

        print(
            "TOP 3 MODEL PREDICTIONS"
        )

        print(
            "=============================="
        )


        for index in top_indices:

            raw_class = (
                class_names[index]
            )


            raw_confidence = float(
                predictions[0][index]
            ) * 100


            print(
                raw_class,
                "->",
                round(
                    raw_confidence,
                    2
                ),
                "%"
            )


            if "___" in raw_class:

                top_crop, top_disease = (
                    raw_class.split(
                        "___",
                        1
                    )
                )

            else:

                top_crop = raw_class

                top_disease = "Unknown"


            top_crop = (
                top_crop
                .replace("_", " ")
                .replace(",", "")
                .strip()
            )


            top_disease = (
                top_disease
                .replace("_", " ")
                .strip()
            )


            top_predictions.append({

                "crop": top_crop,

                "disease": top_disease,

                "confidence": round(
                    raw_confidence,
                    2
                )
            })


        print(
            "==============================\n"
        )


        # ======================================
        # LOW CONFIDENCE SAFETY CHECK
        # ======================================

        CONFIDENCE_THRESHOLD = 70.0


        if confidence < CONFIDENCE_THRESHOLD:

            return render_template(

                "uncertain_result.html",

                confidence=round(
                    confidence,
                    2
                ),

                top_predictions=(
                    top_predictions
                )
            )


        # ======================================
        # SEPARATE CROP + DISEASE
        # ======================================

        if "___" in predicted_class:

            crop_name, disease_name = (
                predicted_class.split(
                    "___",
                    1
                )
            )

        else:

            crop_name = predicted_class

            disease_name = "Unknown"


        # ======================================
        # CLEAN CROP NAME
        # ======================================

        crop_name = (
            crop_name
            .replace("_", " ")
            .replace(",", "")
            .strip()
        )


        # ======================================
        # CLEAN DISEASE NAME
        # ======================================

        disease_name = (
            disease_name
            .replace("_", " ")
            .strip()
        )


        # ======================================
        # HEALTH STATUS
        # ======================================

        is_healthy = (
            disease_name.lower()
            == "healthy"
        )


        if is_healthy:

            status = "Healthy"

        else:

            status = "Disease Detected"


        # ======================================
        # SAVE DISEASE HISTORY
        # ONLY IF USER IS LOGGED IN
        # ======================================

        if "user_id" in session:

            try:

                history_record = DiseaseHistory(

                    user_id=session[
                        "user_id"
                    ],

                    crop_name=crop_name,

                    disease_name=disease_name,

                    confidence=round(
                        confidence,
                        2
                    ),

                    status=status,

                    prediction_class=(
                        predicted_class
                    ),

                    scan_source=scan_source
                )


                db.session.add(
                    history_record
                )


                db.session.commit()


                print(
                    "✅ Disease scan saved "
                    "to history"
                )


            except Exception as history_error:

                db.session.rollback()


                print(
                    "DISEASE HISTORY "
                    "SAVE ERROR:",
                    history_error
                )


        # ======================================
        # RESULT PAGE
        # ======================================

        return render_template(

            "result.html",

            crop_name=crop_name,

            disease_name=disease_name,

            confidence=round(
                confidence,
                2
            ),

            status=status,

            is_healthy=is_healthy,

            top_predictions=(
                top_predictions
            ),

            symptoms=(
                disease_info[
                    "symptoms"
                ]
            ),

            organic_treatment=(
                disease_info[
                    "organic"
                ]
            ),

            treatment=(
                disease_info[
                    "treatment"
                ]
            ),

            prevention=(
                disease_info[
                    "prevention"
                ]
            )
        )


    except Exception as error:

        print(
            "PREDICTION ERROR:",
            error
        )


        flash(
            "Could not analyze the image. "
            "Please upload a valid leaf image.",
            "error"
        )


        return redirect(
            url_for(
                "detect"
            )
        )

    
# ==========================================
# PROFILE
# ==========================================
@app.route("/profile")
def profile():

    if "user_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    user = db.session.get(
        User,
        session["user_id"]
    )

    return render_template(
        "profile.html",
        user=user
    )

# ==========================================
# REGISTER
# ==========================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )


    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        mobile = request.form.get(
            "mobile",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        # ======================================
        # REQUIRED FIELDS
        # ======================================

        if (
            not name
            or not mobile
            or not email
            or not password
        ):

            flash(
                "Please fill all required fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # ======================================
        # MOBILE VALIDATION
        # ======================================

        if (
            not mobile.isdigit()
            or len(mobile) != 10
        ):

            flash(
                "Please enter a valid 10-digit mobile number.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # ======================================
        # PASSWORD VALIDATION
        # ======================================

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # ======================================
        # DUPLICATE EMAIL
        # ======================================

        existing_email = User.query.filter_by(
            email=email
        ).first()


        if existing_email:

            flash(
                "Email is already registered. Please login.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # ======================================
        # DUPLICATE MOBILE
        # ======================================

        existing_mobile = User.query.filter_by(
            mobile=mobile
        ).first()


        if existing_mobile:

            flash(
                "Mobile number is already registered.",
                "error"
            )

            return redirect(
                url_for("register")
            )


        # ======================================
        # PASSWORD HASH
        # ======================================

        hashed_password = (
            bcrypt.generate_password_hash(
                password
            ).decode("utf-8")
        )


        # ======================================
        # CREATE USER
        # ======================================

        new_user = User(

            name=name,

            mobile=mobile,

            email=email,

            password=hashed_password
        )


        try:

            db.session.add(
                new_user
            )

            db.session.commit()


            flash(
                "Registration successful! Please login.",
                "success"
            )


            return redirect(
                url_for("login")
            )


        except Exception as error:

            db.session.rollback()

            print(
                "REGISTRATION ERROR:",
                error
            )

            flash(
                "Something went wrong while creating your account.",
                "error"
            )

            return redirect(
                url_for("register")
            )


    return render_template(
        "register.html"
    )

# ==========================================
# LOGIN
# ==========================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    # ======================================
    # USER ALREADY LOGGED IN
    # ======================================

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )


    # ======================================
    # LOGIN FORM SUBMITTED
    # ======================================

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )


        if not email or not password:

            flash(
                "Please enter email and password.",
                "error"
            )

            return redirect(
                url_for("login")
            )


        # ======================================
        # FIND USER
        # ======================================

        user = User.query.filter_by(
            email=email
        ).first()


        # ======================================
        # CHECK PASSWORD
        # ======================================

        if user and bcrypt.check_password_hash(
            user.password,
            password
        ):

            # IMPORTANT:
            # Browser-session login only.
            session.permanent = False


            # Clear old session information
            session.clear()


            # Create fresh login session
            session["user_id"] = user.id

            session["user_name"] = user.name

            session["user_email"] = user.email

            session["preferred_language"] = (
                user.preferred_language
            )


            flash(
                f"Welcome back, {user.name}!",
                "success"
            )


            return redirect(
                url_for("dashboard")
            )


        # ======================================
        # WRONG LOGIN
        # ======================================

        flash(
            "Invalid email or password.",
            "error"
        )


        return redirect(
            url_for("login")
        )


    return render_template(
        "login.html"
    )


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    # ======================================
    # LOGIN REQUIRED
    # ======================================

    if "user_id" not in session:

        flash(
            "Please login to access your dashboard.",
            "error"
        )

        return redirect(
            url_for("login")
        )


    # ======================================
    # GET CURRENT USER
    # ======================================

    user_id = session["user_id"]


    user = db.session.get(
        User,
        user_id
    )


    if not user:

        session.clear()

        flash(
            "User account not found. Please login again.",
            "error"
        )

        return redirect(
            url_for("login")
        )


    # ======================================
    # ACTIVE CROPS
    # ======================================

    active_crops = (
        Crop.query
        .filter_by(
            user_id=user_id,
            status="Active"
        )
        .count()
    )


    # ======================================
    # DISEASE SCANS
    # ======================================

    disease_scans = (
        DiseaseHistory.query
        .filter_by(
            user_id=user_id
        )
        .count()
    )


    # ======================================
    # ACTIVE REMINDERS
    #
    # Pending crop activities
    # ======================================

    active_reminders = (
        CropActivity.query
        .filter_by(
            user_id=user_id,
            status="Pending"
        )
        .count()
    )


    # ======================================
    # UPCOMING ACTIVITIES
    #
    # Pending activities from today onward
    # ======================================

    today = datetime.now().date()


    upcoming_activities = (
        CropActivity.query
        .filter(
            CropActivity.user_id == user_id,
            CropActivity.status == "Pending",
            CropActivity.due_date >= today
        )
        .count()
    )


    # ======================================
    # RENDER DASHBOARD
    # ======================================

    return render_template(

        "dashboard.html",

        user=user,

        user_name=user.name,

        active_crops=active_crops,

        upcoming_activities=upcoming_activities,

        disease_scans=disease_scans,

        active_reminders=active_reminders
    )



# ==========================================
# CROP SCHEDULER
# ==========================================

@app.route(
    "/crop-scheduler",
    methods=["GET", "POST"]
)
def crop_scheduler():

    if "user_id" not in session:

        flash(
            "Please login to create a crop schedule.",
            "error"
        )

        return redirect(
            url_for("login")
        )


    if request.method == "POST":

        try:

            # ======================================
            # FORM DATA
            # ======================================

            crop_name = request.form.get(
                "crop_name",
                ""
            ).strip()

            variety = request.form.get(
                "variety",
                ""
            ).strip()

            planting_date = parse_optional_date(
                request.form.get(
                    "planting_date"
                )
            )

            farm_name = request.form.get(
                "farm_name",
                ""
            ).strip()

            area = request.form.get(
                "area",
                ""
            ).strip()

            soil_type = request.form.get(
                "soil_type",
                ""
            )

            irrigation_type = request.form.get(
                "irrigation_type",
                ""
            )

            crop_stage = request.form.get(
                "crop_stage",
                ""
            )

            last_watering_date = (
                parse_optional_date(
                    request.form.get(
                        "last_watering_date"
                    )
                )
            )

            last_compost_date = (
                parse_optional_date(
                    request.form.get(
                        "last_compost_date"
                    )
                )
            )

            last_fertilizer_date = (
                parse_optional_date(
                    request.form.get(
                        "last_fertilizer_date"
                    )
                )
            )

            last_inspection_date = (
                parse_optional_date(
                    request.form.get(
                        "last_inspection_date"
                    )
                )
            )

            notify_me = (
                request.form.get(
                    "notify_me"
                )
                == "yes"
            )


            # ======================================
            # VALIDATION
            # ======================================

            if not crop_name:

                flash(
                    "Crop name is required.",
                    "error"
                )

                return redirect(
                    url_for(
                        "crop_scheduler"
                    )
                )


            if not planting_date:

                flash(
                    "Planting date is required.",
                    "error"
                )

                return redirect(
                    url_for(
                        "crop_scheduler"
                    )
                )


            # ======================================
            # CROP RULES
            # ======================================

            rule = get_crop_rule(
                crop_name
            )


            # ======================================
            # BASE DATES
            # ======================================

            watering_base = (
                last_watering_date
                or planting_date
            )

            compost_base = (
                last_compost_date
                or planting_date
            )

            fertilizer_base = (
                last_fertilizer_date
                or planting_date
            )

            inspection_base = (
                last_inspection_date
                or planting_date
            )


            # ======================================
            # NEXT DATES
            # ======================================

            next_watering_date = (
                watering_base
                + timedelta(
                    days=rule[
                        "watering_days"
                    ]
                )
            )

            next_compost_date = (
                compost_base
                + timedelta(
                    days=rule[
                        "compost_days"
                    ]
                )
            )

            next_fertilizer_date = (
                fertilizer_base
                + timedelta(
                    days=rule[
                        "fertilizer_days"
                    ]
                )
            )

            next_inspection_date = (
                inspection_base
                + timedelta(
                    days=rule[
                        "inspection_days"
                    ]
                )
            )

            expected_harvest_date = (
                planting_date
                + timedelta(
                    days=rule[
                        "harvest_days"
                    ]
                )
            )


            # ======================================
            # SAVE SCHEDULE
            # ======================================

            schedule = CropSchedule(

                user_id=session[
                    "user_id"
                ],

                crop_name=crop_name,

                variety=variety,

                planting_date=planting_date,

                farm_name=farm_name,

                area=area,

                soil_type=soil_type,

                irrigation_type=irrigation_type,

                crop_stage=crop_stage,

                last_watering_date=(
                    last_watering_date
                ),

                last_compost_date=(
                    last_compost_date
                ),

                last_fertilizer_date=(
                    last_fertilizer_date
                ),

                last_inspection_date=(
                    last_inspection_date
                ),

                next_watering_date=(
                    next_watering_date
                ),

                next_compost_date=(
                    next_compost_date
                ),

                next_fertilizer_date=(
                    next_fertilizer_date
                ),

                next_inspection_date=(
                    next_inspection_date
                ),

                expected_harvest_date=(
                    expected_harvest_date
                ),

                notify_me=notify_me,

                status="Active"
            )


            db.session.add(
                schedule
            )

            db.session.commit()


            # ======================================
            # CREATE ACTIVITY HISTORY
            # ======================================

            create_initial_crop_activities(
                schedule
            )


            # ======================================
            # CONFIRMATION EMAIL
            # ======================================

            if notify_me:

                user = db.session.get(
                    User,
                    session["user_id"]
                )

                try:

                    send_schedule_confirmation_email(
                        user,
                        schedule
                    )

                    print(
                        "✅ Schedule confirmation "
                        "email sent"
                    )

                except Exception as mail_error:

                    print(
                        "SCHEDULE MAIL ERROR:",
                        mail_error
                    )


            # ======================================
            # RESULT
            # ======================================

            return render_template(
                "schedule_result.html",
                schedule=schedule
            )


        except Exception as error:

            db.session.rollback()

            print(
                "SCHEDULE ERROR:",
                error
            )

            flash(
                "Could not generate schedule.",
                "error"
            )

            return redirect(
                url_for(
                    "crop_scheduler"
                )
            )


    return render_template(
        "add_crop.html"
    )


# ==========================================
# MANAGE CROPS
# ==========================================

@app.route("/crops")
def manage_crops():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    crops = (

        CropSchedule.query

        .filter_by(
            user_id=session[
                "user_id"
            ]
        )

        .order_by(
            CropSchedule.created_at.desc()
        )

        .all()
    )


    for crop in crops:

        ensure_schedule_activities(
            crop
        )


    return render_template(
        "manage_crops.html",
        crops=crops
    )


# ==========================================
# CROP HISTORY
# ==========================================

@app.route(
    "/crops/<int:crop_id>/history"
)
def crop_history(crop_id):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    crop = CropSchedule.query.filter_by(

        id=crop_id,

        user_id=session[
            "user_id"
        ]

    ).first()


    if not crop:

        flash(
            "Crop not found.",
            "error"
        )

        return redirect(
            url_for(
                "manage_crops"
            )
        )


    ensure_schedule_activities(
        crop
    )


    activities = (

        CropActivity.query

        .filter_by(
            schedule_id=crop.id,
            user_id=session[
                "user_id"
            ]
        )

        .order_by(
            CropActivity.due_date.asc()
        )

        .all()
    )


    today = datetime.now().date()


    return render_template(

        "crop_history.html",

        crop=crop,

        activities=activities,

        today=today
    )

# ==========================================
# REMINDERS PAGE
# ==========================================

@app.route("/reminders")
def reminders():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    today = datetime.now().date()

    activities = (

        CropActivity.query

        .filter_by(
            user_id=session["user_id"],
            status="Pending"
        )

        .order_by(
            CropActivity.due_date.asc()
        )

        .all()
    )

    due_today = []

    overdue = []

    upcoming = []

    for activity in activities:

        if activity.due_date < today:

            overdue.append(activity)

        elif activity.due_date == today:

            due_today.append(activity)

        else:

            upcoming.append(activity)

    return render_template(
        "reminders.html",
        due_today=due_today,
        overdue=overdue,
        upcoming=upcoming,
        today=today
    )

# ==========================================
# PLANTING TECHNIQUES
# GROQ + POSTGRESQL + LANGUAGE CACHE
# ==========================================

@app.route(
    "/planting-techniques",
    methods=["GET"]
)
def planting_techniques():

    # ======================================
    # SUPPORTED LANGUAGES
    # ======================================

    languages = {

        "en": {
            "name": "English",
            "voice": "en-IN"
        },

        "mr": {
            "name": "मराठी",
            "voice": "mr-IN"
        },

        "hi": {
            "name": "हिंदी",
            "voice": "hi-IN"
        }
    }


    # ======================================
    # CURRENT LANGUAGE
    # ======================================

    selected_language = request.args.get(
        "language",
        "en"
    ).strip().lower()


    if selected_language not in languages:

        selected_language = "en"


    # ======================================
    # POPULAR PLANTS
    # ======================================

    popular_plants = [

        {"name": "Tomato", "icon": "🍅"},
        {"name": "Potato", "icon": "🥔"},
        {"name": "Banana", "icon": "🍌"},
        {"name": "Mango", "icon": "🥭"},
        {"name": "Rice", "icon": "🌾"},
        {"name": "Wheat", "icon": "🌾"},
        {"name": "Corn", "icon": "🌽"},
        {"name": "Onion", "icon": "🧅"},
        {"name": "Bell Pepper", "icon": "🫑"},
        {"name": "Cotton", "icon": "🌿"},
        {"name": "Sugarcane", "icon": "🎋"},
        {"name": "Apple", "icon": "🍎"},
        {"name": "Grape", "icon": "🍇"},
        {"name": "Strawberry", "icon": "🍓"},
        {"name": "Soybean", "icon": "🌱"},
        {"name": "Groundnut", "icon": "🥜"},
        {"name": "Chickpea", "icon": "🌱"},
        {"name": "Turmeric", "icon": "🌿"},
        {"name": "Ginger", "icon": "🌿"},
        {"name": "Rose", "icon": "🌹"},
        {"name": "Papaya", "icon": "🌴"},
        {"name": "Coconut", "icon": "🥥"},
        {"name": "Pomegranate", "icon": "🍎"},
        {"name": "Guava", "icon": "🌿"}
    ]


    # ======================================
    # SEARCHED PLANT
    # ======================================

    selected_crop = request.args.get(
        "crop",
        ""
    ).strip()


    selected_technique = None

    error_message = None

    guide_source = None


    # ======================================
    # PAGE OPENED WITHOUT SEARCH
    # ======================================

    if not selected_crop:

        return render_template(

            "planting_techniques.html",

            popular_plants=popular_plants,

            selected_crop="",

            selected_technique=None,

            error_message=None,

            guide_source=None,

            languages=languages,

            selected_language=selected_language,

            voice_language=(
                languages[
                    selected_language
                ]["voice"]
            )
        )


    # ======================================
    # NORMALIZE PLANT NAME
    # ======================================

    normalized_plant = (
        selected_crop
        .lower()
        .strip()
    )


    # ======================================
    # LANGUAGE-SPECIFIC CACHE KEY
    #
    # banana::en
    # banana::mr
    # banana::hi
    # ======================================

    plant_key = (
        f"{normalized_plant}"
        f"::{selected_language}"
    )


    # ======================================
    # CHECK POSTGRESQL CACHE
    # ======================================

    cached_guide = PlantGuide.query.filter_by(
        plant_key=plant_key
    ).first()


    if cached_guide:

        try:

            selected_technique = json.loads(
                cached_guide.guide_json
            )

            guide_source = "Database"


            print(
                "✅ Plant guide loaded from database:",
                plant_key
            )


        except Exception as cache_error:

            print(
                "PLANT CACHE ERROR:",
                cache_error
            )

            selected_technique = None


    # ======================================
    # NOT CACHED → GROQ AI
    # ======================================

    if selected_technique is None:

        try:

            print(
                "🤖 Generating plant guide:",
                selected_crop,
                selected_language
            )


            ai_guide = generate_planting_guide(

                selected_crop,

                selected_language
            )


            if not ai_guide:

                error_message = (
                    "Plant guide could not be generated. "
                    "Please check the plant name."
                )


            else:

                selected_technique = ai_guide

                guide_source = "AI Generated"


                # ==================================
                # SAVE LANGUAGE-SPECIFIC GUIDE
                # ==================================

                new_guide = PlantGuide(

                    plant_key=plant_key,

                    plant_name=(
                        ai_guide.get(
                            "name",
                            selected_crop
                        )
                    ),

                    category=(
                        ai_guide.get(
                            "category",
                            "Other"
                        )
                    ),

                    guide_json=json.dumps(
                        ai_guide,
                        ensure_ascii=False
                    )
                )


                db.session.add(
                    new_guide
                )

                db.session.commit()


                print(
                    "✅ Plant guide saved:",
                    plant_key
                )


        except Exception as ai_error:

            db.session.rollback()


            print(
                "PLANT AI ERROR:",
                ai_error
            )


            error_message = (
                "AI guide is temporarily unavailable. "
                "Please try again."
            )


    # ======================================
    # RENDER PAGE
    # ======================================

    return render_template(

        "planting_techniques.html",

        popular_plants=popular_plants,

        selected_crop=selected_crop,

        selected_technique=selected_technique,

        error_message=error_message,

        guide_source=guide_source,

        languages=languages,

        selected_language=selected_language,

        voice_language=(
            languages[
                selected_language
            ]["voice"]
        )
    )

# ==========================================
# PLANTING GUIDE TEXT TO SPEECH
# ==========================================

@app.route(
    "/planting-guide-voice",
    methods=["POST"]
)
def planting_guide_voice():

    try:

        # ======================================
        # GET DATA FROM FRONTEND
        # ======================================

        data = request.get_json(
            silent=True
        ) or {}


        text = str(
            data.get(
                "text",
                ""
            )
        ).strip()


        language = str(
            data.get(
                "language",
                "en"
            )
        ).strip().lower()


        # ======================================
        # VALIDATE TEXT
        # ======================================

        if not text:

            return jsonify({

                "success": False,

                "message": "No guide text received."

            }), 400


        # ======================================
        # LANGUAGE MAP
        # ======================================

        language_map = {

            "en": "en",

            "mr": "mr",

            "hi": "hi"

        }


        tts_language = language_map.get(
            language,
            "en"
        )


        # ======================================
        # CREATE TEMPORARY MP3
        # ======================================

        temp_file = tempfile.NamedTemporaryFile(

            suffix=".mp3",

            delete=False

        )


        temp_path = temp_file.name

        temp_file.close()


        # ======================================
        # GENERATE AUDIO
        # ======================================

        tts = gTTS(

            text=text,

            lang=tts_language,

            slow=False

        )


        tts.save(
            temp_path
        )


        # ======================================
        # SEND AUDIO TO BROWSER
        # ======================================

        response = send_file(

            temp_path,

            mimetype="audio/mpeg",

            as_attachment=False

        )


        # ======================================
        # DELETE TEMP FILE AFTER RESPONSE
        # ======================================

        @response.call_on_close
        def remove_temp_file():

            try:

                if os.path.exists(
                    temp_path
                ):

                    os.remove(
                        temp_path
                    )

            except Exception as cleanup_error:

                print(
                    "TTS TEMP FILE CLEANUP ERROR:",
                    cleanup_error
                )


        return response


    except Exception as error:

        print(
            "PLANTING VOICE ERROR:",
            error
        )


        return jsonify({

            "success": False,

            "message": (
                "Could not generate voice."
            )

        }), 500

# ==========================================
# DISEASE HISTORY PAGE
# ==========================================

@app.route("/disease-history")
def disease_history():

    if "user_id" not in session:

        flash(
            "Please login to view disease history.",
            "error"
        )

        return redirect(
            url_for("login")
        )


    scans = (

        DiseaseHistory.query

        .filter_by(
            user_id=session["user_id"]
        )

        .order_by(
            DiseaseHistory.scanned_at.desc()
        )

        .all()
    )


    total_scans = len(
        scans
    )


    diseased_scans = sum(

        1

        for scan in scans

        if scan.status
        == "Disease Detected"
    )


    healthy_scans = sum(

        1

        for scan in scans

        if scan.status
        == "Healthy"
    )


    return render_template(

        "disease_history.html",

        scans=scans,

        total_scans=total_scans,

        diseased_scans=diseased_scans,

        healthy_scans=healthy_scans
    )

# ==========================================
# MARK ACTIVITY AS DONE
# ==========================================

@app.route(
    "/crop-activity/<int:activity_id>/done",
    methods=["POST"]
)
def mark_crop_activity_done(
    activity_id
):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    activity = CropActivity.query.filter_by(

        id=activity_id,

        user_id=session[
            "user_id"
        ]

    ).first()


    if not activity:

        flash(
            "Activity not found.",
            "error"
        )

        return redirect(
            url_for(
                "manage_crops"
            )
        )


    if activity.status == "Completed":

        return redirect(
            url_for(
                "crop_history",
                crop_id=(
                    activity.schedule_id
                )
            )
        )


    schedule = CropSchedule.query.filter_by(

        id=activity.schedule_id,

        user_id=session[
            "user_id"
        ]

    ).first()


    if not schedule:

        flash(
            "Crop schedule not found.",
            "error"
        )

        return redirect(
            url_for(
                "manage_crops"
            )
        )


    today = datetime.now().date()


    activity.status = "Completed"

    activity.completed_date = today


    rule = get_crop_rule(
        schedule.crop_name
    )


    new_due_date = None


    # ======================================
    # WATERING
    # ======================================

    if (
        activity.activity_type
        == "Watering"
    ):

        new_due_date = (
            today
            + timedelta(
                days=rule[
                    "watering_days"
                ]
            )
        )

        schedule.last_watering_date = (
            today
        )

        schedule.next_watering_date = (
            new_due_date
        )


    # ======================================
    # COMPOST
    # ======================================

    elif (
        activity.activity_type
        == "Compost"
    ):

        new_due_date = (
            today
            + timedelta(
                days=rule[
                    "compost_days"
                ]
            )
        )

        schedule.last_compost_date = (
            today
        )

        schedule.next_compost_date = (
            new_due_date
        )


    # ======================================
    # FERTILIZER
    # ======================================

    elif (
        activity.activity_type
        == "Fertilizer"
    ):

        new_due_date = (
            today
            + timedelta(
                days=rule[
                    "fertilizer_days"
                ]
            )
        )

        schedule.last_fertilizer_date = (
            today
        )

        schedule.next_fertilizer_date = (
            new_due_date
        )


    # ======================================
    # INSPECTION
    # ======================================

    elif (
        activity.activity_type
        == "Inspection"
    ):

        new_due_date = (
            today
            + timedelta(
                days=rule[
                    "inspection_days"
                ]
            )
        )

        schedule.last_inspection_date = (
            today
        )

        schedule.next_inspection_date = (
            new_due_date
        )


    # ======================================
    # HARVEST
    # ======================================

    elif (
        activity.activity_type
        == "Harvest"
    ):

        schedule.status = "Completed"

        schedule.notify_me = False


    # ======================================
    # CREATE NEXT RECURRING ACTIVITY
    # ======================================

    if new_due_date:

        next_activity = CropActivity(

            schedule_id=schedule.id,

            user_id=session[
                "user_id"
            ],

            activity_type=(
                activity.activity_type
            ),

            due_date=new_due_date,

            status="Pending"
        )

        db.session.add(
            next_activity
        )


    db.session.commit()


    flash(
        (
            f"{activity.activity_type} "
            "marked as completed."
        ),
        "success"
    )


    return redirect(
        url_for(
            "crop_history",
            crop_id=schedule.id
        )
    )



# ==========================================
# ARCHIVE CROP
# ==========================================

@app.route(
    "/crops/<int:crop_id>/archive",
    methods=["POST"]
)
def archive_crop(crop_id):

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    crop = CropSchedule.query.filter_by(

        id=crop_id,

        user_id=session[
            "user_id"
        ]

    ).first()


    if not crop:

        flash(
            "Crop not found.",
            "error"
        )

        return redirect(
            url_for(
                "manage_crops"
            )
        )


    crop.status = "Archived"

    crop.notify_me = False


    db.session.commit()


    flash(
        (
            "Crop archived successfully. "
            "Email reminders are stopped."
        ),
        "success"
    )


    return redirect(
        url_for(
            "manage_crops"
        )
    )


# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    # Remove complete login session
    session.clear()


    flash(
        "You have been logged out successfully.",
        "success"
    )


    return redirect(
        url_for("home")
    )


# ==========================================
# FORGOT PASSWORD
# ==========================================

@app.route(
    "/forgot-password",
    methods=["GET", "POST"]
)
def forgot_password():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()


        if not email:

            flash(
                "Please enter your registered email.",
                "error"
            )

            return redirect(
                url_for("forgot_password")
            )


        # ======================================
        # FIND USER
        # ======================================

        user = User.query.filter_by(
            email=email
        ).first()


        if not user:

            flash(
                "No account found with this email.",
                "error"
            )

            return redirect(
                url_for("forgot_password")
            )


        # ======================================
        # CREATE RESET TOKEN
        # ======================================

        token = serializer.dumps(

            user.email,

            salt="password-reset-salt"
        )


        # ======================================
        # CREATE RESET LINK
        # ======================================

        reset_link = url_for(

            "reset_password",

            token=token,

            _external=True
        )


        # ======================================
        # CREATE EMAIL
        # ======================================

        message = Message(

            subject="PlantCare Password Reset",

            sender=app.config[
                "MAIL_USERNAME"
            ],

            recipients=[
                user.email
            ]
        )


        message.body = f"""
Hello {user.name},

We received a request to reset your PlantCare password.

Click the link below to create a new password:

{reset_link}

This password reset link will expire in 30 minutes.

If you did not request a password reset,
you can safely ignore this email.

Regards,
PlantCare Team 🌱
"""


        # ======================================
        # SEND EMAIL
        # ======================================

        try:

            mail.send(
                message
            )


            flash(
                "Password reset link has been sent to your email.",
                "success"
            )


        except Exception as error:

            print(
                "MAIL ERROR:",
                error
            )


            flash(
                "Could not send reset email. Please check mail settings.",
                "error"
            )


        return redirect(
            url_for("forgot_password")
        )


    return render_template(
        "forgot_password.html"
    )


# ==========================================
# RESET PASSWORD
# ==========================================

@app.route(
    "/reset-password/<token>",
    methods=["GET", "POST"]
)
def reset_password(token):

    # ======================================
    # VERIFY TOKEN
    # ======================================

    try:

        email = serializer.loads(

            token,

            salt="password-reset-salt",

            max_age=1800
        )


    except SignatureExpired:

        flash(
            "Password reset link has expired. Please request a new one.",
            "error"
        )


        return redirect(
            url_for("forgot_password")
        )


    except BadSignature:

        flash(
            "Invalid password reset link.",
            "error"
        )


        return redirect(
            url_for("forgot_password")
        )


    # ======================================
    # FIND USER
    # ======================================

    user = User.query.filter_by(
        email=email
    ).first()


    if not user:

        flash(
            "User account not found.",
            "error"
        )


        return redirect(
            url_for("forgot_password")
        )


    # ======================================
    # NEW PASSWORD SUBMITTED
    # ======================================

    if request.method == "POST":

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        # ==================================
        # EMPTY VALIDATION
        # ==================================

        if not password or not confirm_password:

            flash(
                "Please fill both password fields.",
                "error"
            )


            return redirect(

                url_for(

                    "reset_password",

                    token=token
                )
            )


        # ==================================
        # PASSWORD LENGTH
        # ==================================

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )


            return redirect(

                url_for(

                    "reset_password",

                    token=token
                )
            )


        # ==================================
        # PASSWORD MATCH
        # ==================================

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )


            return redirect(

                url_for(

                    "reset_password",

                    token=token
                )
            )


        # ==================================
        # HASH NEW PASSWORD
        # ==================================

        hashed_password = (
            bcrypt.generate_password_hash(
                password
            ).decode("utf-8")
        )


        user.password = hashed_password


        try:

            db.session.commit()


            # Old login session clear
            session.clear()


            flash(
                "Password reset successful! Please login with your new password.",
                "success"
            )


            return redirect(
                url_for("login")
            )


        except Exception as error:

            db.session.rollback()


            print(
                "PASSWORD RESET ERROR:",
                error
            )


            flash(
                "Could not reset password. Please try again.",
                "error"
            )


            return redirect(

                url_for(

                    "reset_password",

                    token=token
                )
            )


    return render_template(

        "reset_password.html",

        token=token
    )


# ==========================================
# CREATE DATABASE TABLES
# ==========================================

with app.app_context():

    db.create_all()


# ==========================================
# BACKGROUND EMAIL REMINDER SCHEDULER
# ==========================================

scheduler = BackgroundScheduler(
    timezone="Asia/Kolkata"
)

scheduler.add_job(
    scheduled_reminder_job,
    trigger="cron",
    hour=8,
    minute=0,
    id="daily_crop_reminders",
    replace_existing=True
)

# ==========================================
# PLANTCARE AI FARMER CHATBOT
# ==========================================

@app.route(
    "/api/farmer-chat",
    methods=["POST"]
)
def farmer_chat():

    try:

        # Login required
        if "user_id" not in session:

            return jsonify({
                "success": False,
                "message": "Please login first."
            }), 401


        data = request.get_json(
            silent=True
        ) or {}


        question = str(
            data.get("question", "")
        ).strip()


        language = str(
            data.get("language", "en")
        ).strip().lower()


        if not question:

            return jsonify({
                "success": False,
                "message": "Please enter your farming question."
            }), 400


        if language not in [
            "en",
            "mr",
            "hi"
        ]:

            language = "en"


        # ======================================
        # GET AI ANSWER
        # ======================================

        answer = get_farmer_ai_response(
            question=question,
            language=language
        )


        if not answer:

            return jsonify({
                "success": False,
                "message": "AI could not generate a response."
            }), 500


        # ======================================
        # SAVE CHAT IN POSTGRESQL
        # ======================================

        chat = ChatHistory(

            user_id=session["user_id"],

            question=question,

            answer=answer,

            language=language
        )


        db.session.add(chat)

        db.session.commit()


        return jsonify({

            "success": True,

            "answer": answer,

            "language": language

        })


    except Exception as error:

        db.session.rollback()

        print(
            "FARMER CHAT ERROR:",
            error
        )


        return jsonify({

            "success": False,

            "message": (
                "PlantCare AI is temporarily unavailable. "
                "Please try again."
            )

        }), 500

    # ==========================================
# LOAD AI CHAT HISTORY
# ==========================================

@app.route(
    "/api/farmer-chat/history",
    methods=["GET"]
)
def farmer_chat_history():

    try:

        if "user_id" not in session:

            return jsonify({
                "success": False,
                "message": "Please login first."
            }), 401


        chats = (
            ChatHistory.query
            .filter_by(
                user_id=session["user_id"]
            )
            .order_by(
                ChatHistory.created_at.asc()
            )
            .limit(50)
            .all()
        )


        history = []


        for chat in chats:

            history.append({

                "id": chat.id,

                "question": chat.question,

                "answer": chat.answer,

                "language": chat.language,

                "created_at":
                    chat.created_at.strftime(
                        "%d %b %Y, %I:%M %p"
                    )
            })


        return jsonify({

            "success": True,

            "history": history

        })


    except Exception as error:

        print(
            "LOAD CHAT HISTORY ERROR:",
            error
        )


        return jsonify({
            "success": False,
            "message": "Could not load chat history."
        }), 500



# ==========================================
# CLEAR AI CHAT HISTORY
# ==========================================

@app.route(
    "/api/farmer-chat/history",
    methods=["DELETE"]
)
def clear_farmer_chat_history():

    try:

        if "user_id" not in session:

            return jsonify({
                "success": False,
                "message": "Please login first."
            }), 401


        ChatHistory.query.filter_by(
            user_id=session["user_id"]
        ).delete()


        db.session.commit()


        return jsonify({

            "success": True,

            "message": "Chat history cleared."

        })


    except Exception as error:

        db.session.rollback()


        print(
            "CLEAR CHAT HISTORY ERROR:",
            error
        )


        return jsonify({
            "success": False,
            "message": "Could not clear chat history."
        }), 500

# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    # Flask debug mode uses a reloader.
    # Start the scheduler only in the real serving process,
    # not in the parent reloader process.
    if (
        os.environ.get("WERKZEUG_RUN_MAIN") == "true"
        or not app.debug
    ):

        if not scheduler.running:
            scheduler.start()

    app.run(
        debug=True
    )
