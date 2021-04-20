import time
import pandas as pd
from collections import OrderedDict
# from IPython.display import display, clear_output


class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.show_data = []
        self.run_start_time = None

        self.test_loss = 0
        self.test_accuracy = 0

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

    def end_epoch(self):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['train_loss'] = loss
        results["train_acc"] = accuracy
        results['test_loss'] = self.test_loss
        results["test_acc"] = self.test_accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items():
            results[k] = v

        self.run_data.append(results)

        epoch_interval = 25
        if self.epoch_count % epoch_interval == 0:
            # show_data = [run for run in self.run_data if run['epoch'] % epoch_interval == 0]
            # df = pd.DataFrame(show_data)
            # clear_output(wait=True)
            # display(df)
            print(f"epoch:{self.epoch_count}", end='\t')
            print(f"train_acc:{accuracy:.3f}", end='\t')
            print(f"test_acc:{self.test_accuracy:.3f}")

    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    def track_test_loss(self, loss, test_data):
        self.test_loss = loss.item()

    def track_test_accuracy(self, preds, labels):
        self.test_accuracy = self._get_num_correct(preds, labels) / len(labels)

    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')

#         with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
#             json.dump(self.run_data, f, ensure_ascii=False, indent=4)
