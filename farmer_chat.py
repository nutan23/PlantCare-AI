# ==========================================
# PLANTCARE AI FARMER ASSISTANT
# ==========================================

import os

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
# LANGUAGE INSTRUCTIONS
# ==========================================

LANGUAGE_PROMPTS = {

    "en": """
Answer in simple English.
Use language that an Indian farmer can understand.
""",

    "mr": """
उत्तर फक्त मराठी भाषेत द्या.
नैसर्गिक आणि सोपी मराठी वापरा.
शक्यतो देवनागरी लिपी वापरा.
शेतकऱ्याला सहज समजेल अशा पद्धतीने उत्तर द्या.
""",

    "hi": """
उत्तर केवल हिंदी भाषा में दें।
सरल और प्राकृतिक हिंदी का उपयोग करें।
देवनागरी लिपि का उपयोग करें।
किसान को आसानी से समझ आने वाली भाषा में उत्तर दें।
"""
}


# ==========================================
# FARMER AI CHAT
# ==========================================

def get_farmer_ai_response(
    question,
    language="en"
):

    question = str(
        question
    ).strip()


    if not question:

        return None


    if language not in LANGUAGE_PROMPTS:

        language = "en"


    language_instruction = (
        LANGUAGE_PROMPTS[
            language
        ]
    )


    system_prompt = f"""
You are PlantCare AI Farmer Assistant.

You help farmers and agriculture students with:

- crop cultivation
- planting
- crop scheduling
- watering
- irrigation
- soil management
- compost
- fertilizer
- plant nutrition
- plant diseases
- disease prevention
- crop symptoms
- harvesting
- weeds
- weather-related general crop care
- organic farming
- farm management
- fruits
- vegetables
- cereals
- pulses
- flowers
- agricultural plants

LANGUAGE:

{language_instruction}

IMPORTANT RULES:

1. Give practical and easy-to-understand answers.

2. Keep normal answers concise.

3. When the farmer asks "how to grow",
   "how to plant", or asks for a process,
   give clear numbered steps.

4. Do not invent pesticide dosages.

5. Do not provide dangerous chemical
   mixing instructions.

6. For pesticides or chemicals, advise
   following the product label and local
   agricultural recommendations.

7. If the answer depends strongly on
   location, weather, crop variety or soil,
   mention that clearly.

8. Do not pretend to know the user's
   current weather or soil condition.

9. If the question is completely unrelated
   to farming, agriculture, plants or crops,
   politely explain that you are a farming
   assistant.

10. Do not use markdown tables.

11. Avoid unnecessarily long answers.
"""


    try:

        response = (
            client.chat.completions.create(

                model="openai/gpt-oss-20b",

                messages=[

                    {
                        "role": "system",
                        "content": system_prompt
                    },

                    {
                        "role": "user",
                        "content": question
                    }

                ],

                temperature=0.35,

                max_completion_tokens=1000
            )
        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )


        if not answer:

            return None


        return answer.strip()


    except Exception as error:

        print(
            "FARMER AI ERROR:",
            error
        )

        raise