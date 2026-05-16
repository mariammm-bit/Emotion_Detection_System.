from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self):
        self.objects = OrderedDict()
        self.next_id = 0

    def update(self, rects):
        objects = {}

        for rect in rects:
            x, y, w, h = rect
            cx = int(x + w/2)
            cy = int(y + h/2)

            objects[self.next_id] = (cx, cy)
            self.next_id += 1

        self.objects = objects
        return self.objects