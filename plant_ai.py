# ==========================================
# GROQ AI PLANTING GUIDE
# MULTI-LANGUAGE
# ==========================================

import os
import json

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()


GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY not found in .env file"
    )


client = Groq(
    api_key=GROQ_API_KEY
)


# ==========================================
# LANGUAGE MAP
# ==========================================

LANGUAGE_NAMES = {

    "en": "English",

    "mr": "Marathi",

    "hi": "Hindi"
}


# ==========================================
# GENERATE PLANT GUIDE
# ==========================================

def generate_planting_guide(
    plant_name,
    language="en"
):

    plant_name = plant_name.strip()


    language_name = LANGUAGE_NAMES.get(
        language,
        "English"
    )


    prompt = f"""
You are an agricultural farming assistant.

Plant or crop:
{plant_name}

Generate a complete practical planting guide.

OUTPUT LANGUAGE:
{language_name}

IMPORTANT LANGUAGE RULE:

All user-visible VALUES must be written in
{language_name}.

Keep JSON KEY names in English exactly as provided.

For Marathi:
use natural Marathi written in Devanagari.

For Hindi:
use natural Hindi written in Devanagari.

For English:
use simple Indian English suitable for farmers.

Return ONLY valid JSON.

Do not use markdown.
Do not use ```json.
Do not write anything outside JSON.


Use exactly this JSON structure:

{{
    "name": "Plant name",

    "category": "Plant category",

    "season": "Suitable planting season",

    "climate": "Suitable climate",

    "soil": "Suitable soil and drainage",

    "soil_ph": "Suitable soil pH",

    "spacing": "Plant and row spacing",

    "planting_depth": "Planting depth",

    "watering": "Watering guidance",

    "fertilizer": "Compost and fertilizer guidance",

    "sunlight": "Sunlight requirement",

    "care": "General crop care",

    "duration": "Approximate growing duration",

    "harvest": "Harvest signs and method",

    "common_problems": [
        "Problem 1",
        "Problem 2",
        "Problem 3"
    ],

    "steps": [
        {{
            "title": "Step title",
            "description": "Detailed practical instruction"
        }}
    ]
}}


RULES:

1. Provide 12 to 15 practical steps.

2. Start from:
site selection,
soil preparation,
seed / seedling / planting material selection.

3. Include:
planting,
spacing,
first watering,
regular watering,
compost,
fertilizer,
weed control,
crop inspection,
growth-stage care,
harvesting.

4. Give practical but general agricultural guidance.

5. Do not give dangerous pesticide mixing instructions.

6. Do not invent chemical pesticide dosages.

7. If advice depends heavily on local soil,
climate or variety, clearly say so.

8. If "{plant_name}" is not a real plant,
crop, tree, herb, flower or agricultural plant,
return exactly:

{{
    "error": "Plant not recognized"
}}

Return JSON only.
"""


    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[

            {
                "role": "system",
                "content": (
                    "You are a structured agricultural "
                    "guidance assistant. Return valid JSON only."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        response_format={
            "type": "json_object"
        },

        temperature=0.25,

        max_completion_tokens=3500
    )


    raw_content = (
        response
        .choices[0]
        .message
        .content
    )


    guide = json.loads(
        raw_content
    )


    if guide.get("error"):

        return None


    if not guide.get("steps"):

        return None


    return guide