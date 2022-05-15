import numpy as np
# from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    print(event_acc.Tags())

    if "explore" in path:
        H = 75
    else:
        H = 5
    step = []
    ret = []
    RET = event_acc.Scalars("test_return_mean")
    for x in RET:
        step.append(x[1])
        ret.append(x[2])
    plt.plot(step, ret)
    # print(step, ret)
    plt.savefig("./temp.png")
    
    full_res = np.stack([np.array(step), np.array(ret)])
    import torch
    full_res = torch.tensor(full_res)
    torch.save(full_res, "./torch_logs/{}.pt".format(NAME_OF_EXP))
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


NAME_OF_EXP = "Sep_A2L4_s3"

if __name__ == '__main__':
    log_dir = "/home/lych/IJCAI22/EMC/pymarl/results/tb_logs/explore/none/EMC_sepVDN__2022-04-11_02-05-10"
    from pathlib import Path
    for f in Path(log_dir).rglob('*'):
        print(str(f))
    f = str(f)
    plot_tensorflow_log(f)

# A2L4_s1: /home/lych/IJCAI22/EMC/pymarl/results/tb_logs/explore/none/EMC_simpleVDN__2022-04-10_23-56-26/events.out.tfevents.1649634986.07584409239e

# Sep_A2L4_s1: /home/lych/IJCAI22/EMC/pymarl/results/tb_logs/explore/none/EMC_sepVDN__2022-04-11_00-14-18/events.out.tfevents.1649636058.07584409239e
# Sep_A2L4_s2: results/tb_logs/explore/none/EMC_sepVDN__2022-04-11_01-29-15
# Sep_A2L4_s3: results/tb_logs/explore/none/EMC_sepVDN__2022-04-11_02-05-10
# s4: results/tb_logs/explore/none/EMC_sepVDN__2022-04-11_03-23-09
# s5: results/tb_logs/explore/none/EMC_sepVDN__2022-04-11_06-34-58