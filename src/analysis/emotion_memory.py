from collections import deque

memory = {}

def update_memory(person_id, emotion):
    if person_id not in memory:
        memory[person_id] = deque(maxlen=30)

    memory[person_id].append(emotion)

def get_history(person_id):
    return memory.get(person_id, [])