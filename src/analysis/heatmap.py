import numpy as np
import cv2

heatmap = np.zeros((480,640), dtype=np.float32)

def update_heatmap(x,y,w,h):
    heatmap[y:y+h, x:x+w] += 1

def overlay_heatmap(frame):
    norm = heatmap / (heatmap.max() + 1e-6)
    heat = (norm * 255).astype("uint8")
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    return cv2.addWeighted(frame, 0.7, heat, 0.3, 0)