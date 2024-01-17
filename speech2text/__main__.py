from .listener import MicrophoneListener

if __name__ == "__main__":
    # MicrophoneListener()
    print(*MicrophoneListener.get_device_list(), sep="\n")
