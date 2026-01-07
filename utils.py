# utils.py
import time
import threading

class FrameRateLimiter:
    """Simple frame rate limiter for responsive controls"""
    
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.last_frame_time = 0
        self.lock = threading.Lock()
    
    def should_update(self) -> bool:
        """Check if enough time has passed for a new frame"""
        with self.lock:
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_time:
                self.last_frame_time = current_time
                return True
            return False
    
    def update(self):
        """Mark that an update has occurred"""
        with self.lock:
            self.last_frame_time = time.time()