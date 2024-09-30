import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical, Sequence
from dsnplus.utils import preprocess_image
from sklearn.utils import class_weight

class DataGenerator(Sequence):
    def __init__(self, data, n_classes, batch_size, risk_factor, augmentation=False, shuffle=False):
        self.data = data
        self.agumentation = augmentation
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.risk_factor = risk_factor

        self._get_chunks()
        print('data generator loaded')
        print('data size: '+str(len(data)))
        # log.info('shuffle: '+str(self.shuffle))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        rows = self.chunks[index]
        batch_images, batch_labels = self._images_process(rows, self.risk_factor)
        return batch_images, batch_labels

    def _images_process(self, rows, risk_factor):
        batch_images = np.array([]) 
        batch_labels = []

        print("rows", rows.columns)
        for _, row in rows.iterrows():
            
            file_path, label = row['pathname'], row[self.risk_factor]
            try:
                x = preprocess_image(file_path)
                if len(batch_images):
                    batch_images = np.vstack((batch_images, x))
                else:
                    batch_images = x
                batch_labels.append(label)
            except Exception as e:
                print(f"Error processing image with name {file_path}: {e}")
                continue

        return batch_images, to_categorical(np.array(batch_labels), self.n_classes)


    def on_epoch_end(self):
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def get_epoch_num(self):
        return self.cal_chunk_number(len(self.data), self.batch_size)

    def _get_chunks(self):
        self.chunks = np.array_split(self.data, self.cal_chunk_number(len(self.data), self.batch_size))

    def cal_chunk_number(self, total_size, batch_size):
        return total_size // batch_size  if total_size % batch_size == 0 else (total_size // batch_size) + 1

    def class_weights(self):
        labels = []
        for _, row in self.data.iterrows():
            _, label = row[0], row['converted_drusen']
            labels.append(label)
        class_weight_list = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
        cw = dict(zip(np.unique(labels), class_weight_list))
        return cw

    @property
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass