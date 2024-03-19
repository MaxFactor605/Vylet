import tensorflow as tf
import numpy as np


log_dir = "log_big_v7."

MAPS = ["round", "round-grass", "round-r", "round-grass-r", "curvy", "curvy-r", "long", "long-r", "big", "big-r",
            "zig-zag", "zig-zag-r", "plus", "plus-r", "H", "H-r"]


for v in range(1, 14):
    print("Proccessing {}".format(log_dir+str(v)))
    step = 500_000
    train_log_dir = log_dir + '{}/tensorboard/RecurrentPPO_0'.format(v)

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)



    reward = np.load(log_dir + "{}/reward_log.npy".format(v))

    percents = np.load(log_dir + "{}/complete_log.npy".format(v))

    for i, perc_env in enumerate(percents):
        for j, perc in enumerate(perc_env):
            with train_summary_writer.as_default():
                tf.summary.scalar('EvalCompleteness/{}'.format(MAPS[j]), perc, step=step)
                tf.summary.scalar('EvalReward/{}'.format(MAPS[j]), reward[i][j], step= step)
        print("All maps at {} step added".format(step))
            #tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        step+=500_000



    reward_mean = np.mean(reward, axis = 1)
  
    perc_mean = np.mean(percents, axis = 1)
    step = 500_000

    for i, perc in enumerate(perc_mean):
        with train_summary_writer.as_default():
            tf.summary.scalar('EvalCompletenessMean', perc, step=step)
            tf.summary.scalar('EvalRewardMean', reward_mean[i], step= step)
            #tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            print("Mean at {} step added".format(step))
            step+=500_000


