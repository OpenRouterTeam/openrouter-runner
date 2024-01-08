from shared.protocol import create_error_text


def run_in_thread(task, *args, **kwargs):
    import threading
    import time

    """
    Runs a given task in a separate thread and yields control until the task is completed.
    """
    output = [None]
    output_ready = threading.Event()

    def working_thread():
        try:
            output[0] = task(*args, **kwargs)
        except Exception as err:
            output[0] = create_error_text(err)
            print(output[0])
        finally:
            output_ready.set()

    threading.Thread(target=working_thread).start()

    # Continuously yield control until the thread is done
    while not output_ready.is_set():
        yield " "  # Yield an empty space
        time.sleep(0.1)  # Adjust sleep time as needed

    # Yield the final output
    yield output[0]
