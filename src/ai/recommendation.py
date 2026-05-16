import webbrowser
import time

# متغير عشان السيستم ميفتحش الأغنية 100 مرة في الثانية
last_played = 0 

def recommend(emotion):
    global last_played
    current_time = time.time()
    
    # لاحظي إني خليت أول حرف كابيتال عشان يطابق التعديل اللي فوق
    if emotion == "Sad":
        # لو عدى 30 ثانية على آخر مرة شغل فيها الأغنية عشان ميزعجكيش
        if current_time - last_played > 30:
            print("System Triggered: Opening relaxing music...")
            # الكود ده هيفتح المتصفح ويشغل فيديو مزيكا هادية
            webbrowser.open("https://www.youtube.com/watch?v=lFcSrYw-ARY")
            last_played = current_time
        return "Playing relaxing music... 🎵"
        
    elif emotion == "Angry":
        return "Take a deep breath 🧘"
    elif emotion == "Happy":
        return "Keep it up 😄"
    elif emotion == "Fear":
        return "Stay calm, you're safe 💙"
        
    return "Stay focused 👀"