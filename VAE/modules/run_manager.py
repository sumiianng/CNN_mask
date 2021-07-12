import time
import pandas as pd
from collections import OrderedDict
from IPython.display import display, clear_output


class RunManager():
    def __init__(self):
        pass
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.show_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None

    def begin_run(self, run, network, loader):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader

    def end_run(self):
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self, intev):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)

        if self.epoch_count % intev == 0:
            self.show_data.append(results)

            df = pd.DataFrame(self.show_data)
            clear_output(wait=True)
            display(df)

    def track_loss(self, loss):
        self.epoch_loss += loss.item()
