import numpy as np
# from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(log_dir):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 10000,
        'histograms': 1
    }
    
    from pathlib import Path

    for f in Path(log_dir).rglob('*/events.out.*'):
        print(str(f))

    import torch

    full_ress = []
    for f in Path(log_dir).rglob('*/events.out.*'):
        event_acc = EventAccumulator(str(f), tf_size_guidance)
        event_acc.Reload()

        # Show all tags in the log file

        step = []
        ret = []
        RET = event_acc.Scalars("test_return_mean")
        for x in RET:
            step.append(x[1])
            ret.append(x[2])
        plt.plot(step, ret, color="blue")
        print(step)
        full_res = np.stack([np.array(step), np.array(ret)])
        full_res = torch.tensor(full_res)
        full_ress.append(full_res)
    # print(step, ret)
    plt.savefig("./{}.png".format(NAME_OF_EXP))
    
    torch.save(full_ress, "./torch_logs/{}.pt".format(NAME_OF_EXP))
    return

    """training_accuracies =   event_acc.Scalars('training-accuracy')
    validation_accuracies = event_acc.Scalars('validation_accuracy')

    steps = 10
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in xrange(steps):
        y[i, 0] = training_accuracies[i][2] # value
        y[i, 1] = validation_accuracies[i][2]

    plt.plot(x, y[:,0], label='training accuracy')
    plt.plot(x, y[:,1], label='validation accuracy')

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()"""


NAME_OF_EXP = "Multi_10"

if __name__ == '__main__':
    log_path = "/home/lych/IJCAI22/EMC/pymarl/results/tb_logs/naive/multi_step_10"
    plot_tensorflow_log(log_path)
