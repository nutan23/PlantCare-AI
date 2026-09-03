# ==========================================
# CROP SCHEDULING RULES
# ==========================================

CROP_RULES = {

    "tomato": {
        "watering_days": 3,
        "compost_days": 20,
        "fertilizer_days": 15,
        "inspection_days": 7,
        "harvest_days": 90
    },

    "potato": {
        "watering_days": 4,
        "compost_days": 20,
        "fertilizer_days": 20,
        "inspection_days": 7,
        "harvest_days": 100
    },

    "corn": {
        "watering_days": 5,
        "compost_days": 25,
        "fertilizer_days": 20,
        "inspection_days": 7,
        "harvest_days": 110
    },

    "apple": {
        "watering_days": 7,
        "compost_days": 30,
        "fertilizer_days": 30,
        "inspection_days": 10,
        "harvest_days": 150
    },

    "grape": {
        "watering_days": 5,
        "compost_days": 25,
        "fertilizer_days": 20,
        "inspection_days": 7,
        "harvest_days": 120
    },

    "pepper": {
        "watering_days": 3,
        "compost_days": 20,
        "fertilizer_days": 15,
        "inspection_days": 7,
        "harvest_days": 85
    }
}


DEFAULT_RULE = {
    "watering_days": 4,
    "compost_days": 20,
    "fertilizer_days": 15,
    "inspection_days": 7,
    "harvest_days": 90
}


def get_crop_rule(crop_name):

    crop_name = crop_name.lower().strip()

    return CROP_RULES.get(
        crop_name,
        DEFAULT_RULE
    )