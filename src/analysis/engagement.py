def compute_engagement(attention, emotion_value):
    return round((attention * 0.6) + (emotion_value * 0.4), 2)