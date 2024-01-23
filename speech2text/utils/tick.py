import time


class TickSynchronizer:
    """A context manager, which helps aligning a series of events to
    a time grid.

    Example:
    ```python
    with TickSynchronizer(0.8) as ticker:
        while True:
            time.sleep(random.random())  # wait for [0.0,1.0] sec
            ticker.tick()
            print("tick")

    # This will print "tick" to console with the average frequancy
    #  of 1.25 times/sec (1/0.8 == 1.25)
    ```"""

    def __init__(self, tick_duration_sec: float) -> None:
        self.tick_duration_sec = tick_duration_sec
        self.next_tick = None

    def __enter__(self):
        self.next_tick = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.next_tick = None

    def tick(self):
        self.next_tick += self.tick_duration_sec
        current_time = time.time()
        if self.next_tick > current_time:
            time.sleep(self.next_tick - current_time)


if __name__ == "__main__":
    import random

    print("=== TickSynchronizer demonstration ===")
    print("(press Ctrl+C to stop)")
    print("")

    tick_durastion_sec = 0.05
    sleep_durastion_sec = 0.005
    assert sleep_durastion_sec < tick_durastion_sec

    tick_frequency = 1 / tick_durastion_sec
    ticks_count = 0
    iteration_avg_duration = 0
    try:
        with TickSynchronizer(tick_durastion_sec) as ticker:
            cycle_start = time.time()
            while True:
                iteration_start = time.time()
                time.sleep(random.random() * sleep_durastion_sec * 2)
                ticks_count += 1
                ticker.tick()
                iteration_duration = time.time() - iteration_start

                cycle_duration = time.time() - cycle_start
                iteration_avg_duration = cycle_duration / ticks_count
                print(
                    (
                        "\r"
                        f"iteration_duration = {iteration_duration:.4f}"
                        f"\t   cycle_duration / ticks_count = {iteration_avg_duration:.4f}"
                        f"\t  (both should become close to {tick_durastion_sec:.4f}) "
                        "                               "
                    ),
                    end="",
                )
    except KeyboardInterrupt:
        print()
        avg_error = abs(tick_durastion_sec - iteration_avg_duration)
        avg_error_threshold = 0.005
        assert avg_error <= avg_error_threshold, (
            "avg_error > avg_error_threshold "
            f"({avg_error} > {avg_error_threshold})"
        )
        print()
        print("Everything works fine!")
