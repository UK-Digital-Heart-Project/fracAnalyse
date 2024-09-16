import os
import random
import threading
import time
from tqdm import tqdm

class ParforProgressbar:
    def __init__(self, N_init, message=''):
        self.N = N_init
        self.ipcfile = self.create_ipcfile()
        self.progress_bar = tqdm(total=N_init, desc=message)
        self.timer = threading.Timer(0.5, self.tupdate)
        self.timer.start()

    def create_ipcfile(self):
        for i in range(10):
            f = f"{os.path.basename(__file__).split('.')[0]}{round(random.random() * 1000)}.txt"
            ipcfile = os.path.join(tempfile.gettempdir(), f)
            if not os.path.exists(ipcfile):
                return ipcfile
        raise Exception('Too many temporary files. Clear out tempdir.')

    def close(self):
        if self.timer.is_alive():
            self.timer.cancel()
        if os.path.exists(self.ipcfile):
            os.remove(self.ipcfile)
        self.progress_bar.close()

    def __del__(self):
        self.close()

    @property
    def percent(self):
        if not os.path.exists(self.ipcfile):
            return 0
        with open(self.ipcfile, 'r') as f:
            percent = sum(int(line) for line in f) / self.N
            return max(0, min(1, percent))

    @percent.setter
    def message(self, newMsg):
        self.progress_bar.set_description(newMsg)

    def iterate(self, Nitr=1):
        with open(self.ipcfile, 'a') as f:
            f.write(f"{Nitr}\n")

    def tupdate(self):
        while True:
            self.progress_bar.n = self.percent * self.N
            self.progress_bar.refresh()
            time.sleep(0.5)

# Usage Example:
# pb = ParforProgressbar(100, 'Processing')
# for i in range(100):
#     pb.iterate()
#     time.sleep(0.1)
# pb.close()
