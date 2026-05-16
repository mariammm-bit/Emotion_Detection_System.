emotion_map = {
    "angry": -2,
    "sad": -1,
    "neutral": 0,
    "happy": 1,
    "surprise": 2,
    "fear": -1,
    "disgust": -2
}

def get_trend(history):
    if len(history) < 5:
        return "Analyzing..."

    values = [emotion_map.get(e, 0) for e in history]
    diff = values[-1] - values[0]

    if diff > 1:
        return "Improving 📈"
    elif diff < -1:
        return "Declining 📉"
    return "Stable ➖"