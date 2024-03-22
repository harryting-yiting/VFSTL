from datetime import datetime

def get_current_time_string():
    return datetime.now().strftime('%Y%m%d_%H%M%S')