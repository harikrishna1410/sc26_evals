import time

def echo_hello_world(task_id, sleeptime=0.0):
    """Execute a noop task with optional busy waiting.
    
    Args:
        task_id: Task identifier
        sleeptime: Time to busy wait in seconds
    
    Returns:
        String with task_id
    """
    if sleeptime > 0:
        start = time.perf_counter()
        while (time.perf_counter() - start) < sleeptime:
            pass  # Busy waiting
    return f"Hello World {task_id}"