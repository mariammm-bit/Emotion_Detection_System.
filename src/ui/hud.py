import cv2

def draw_box(frame, x, y, w, h):
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

def draw_text(frame, x, y, emotion, conf, trend, rec):
    # استخدام y- بدل y+ عشان نكتب فوق الوش
    cv2.putText(frame, f"{emotion} ({int(conf*100)}%)",
                (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    cv2.putText(frame, trend,
                (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0),2)

    cv2.putText(frame, rec,
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)