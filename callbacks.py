import numpy as np
import tensorflow as tf


class PerformanceMonitor(tf.keras.callbacks.Callback):

    def __init__(self, monitor='val_elbo', patience=0):
        super().__init__()

        # save configuration
        self.monitor = monitor
        self.patience = patience

        # declare variables
        self.nan_inf = None
        self.best_acc = None
        self.best_clust = None
        self.best_score = None
        self.best_weights = None
        self.stopped_epoch = None
        self.wait = None

    def on_train_begin(self, logs=None):

        # initialize variables
        self.nan_inf = False
        self.best_acc = 0
        self.best_clust = np.inf
        self.best_score = -np.inf
        self.best_weights = None
        self.stopped_epoch = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1

        # reduce vector metrics
        for key in ['clust', 'val_clust']:
            if key in logs.keys() and not isinstance(logs[key], float):
                logs[key] = tf.reduce_sum(tf.cast(tf.greater(logs[key], 0), tf.float32))

        # update string
        update_string = 'epoch {:d}'.format(epoch)
        for key in [k for k in logs.keys() if 'val_' not in k]:
            update_string += ' | {:s} = {:.3g}'.format(key, logs[key])
            if 'val_' + key in logs.keys():
                update_string += ' ({:.3g})'.format(logs['val_' + key])

        # test for NaN and Inf
        if tf.math.is_nan(logs['elbo']) or tf.math.is_inf(logs['elbo']):
            self.nan_inf = True
            self.stopped_epoch = epoch
            self.model.stop_training = True

        # early stopping logic
        current_score = logs.get(self.monitor) or 0.0
        if tf.greater(current_score, self.best_score):
            self.best_acc = logs.get('val_acc')
            self.best_clust = logs.get('val_clust')
            self.best_score = current_score
            self.best_weights = self.model.get_weights()
            self.wait = 0
        elif (logs.get(self.monitor) is not None) and self.patience > 0:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

        # update string
        if self.best_acc is not None and self.best_clust is not None:
            update_string += ' | best: {:.3f} w/ {:g} clusters'.format(self.best_acc, self.best_clust)
        update_string += ' | wait: {:d}/{:d}'.format(self.wait, self.patience)

        # print update
        print('\r' + update_string, end='')

    def on_train_end(self, logs=None):
        if self.nan_inf:
            print('\nEpoch {:d}: NaN or Inf detected!'.format(self.stopped_epoch + 1))
        else:
            print('\nFinished!')
        if self.stopped_epoch > 0:
            print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)


# class Reconstruction(tf.keras.callbacks.Callback):
#
#     def __init__(self, validation_data):
#         super().__init__()
#         self.validation_data = validation_data
#
#     def on_epoch_end(self, epoch, logs=None):
#         for batch in self.validation_data:
#             x = batch['image']
#             px_x = self.model.px_x(x)
#             fig, ax = plt.subplots(nrows=3, ncols=10)
#             mean = tf.reshape(px_x.mean(), [-1] + list(self.model.dim_x))
#             sample = tf.reshape(px_x.sample(), [-1] + list(self.model.dim_x))
#             for c in range(ax.shape[1]):
#                 ax[0, c].imshow(tf.squeeze(x[c]))
#                 ax[1, c].imshow(tf.squeeze(mean[c]))
#                 ax[2, c].imshow(tf.squeeze(sample[c]))
#             fig.savefig('recon.png')
#             plt.close(fig)
