from DataPreparation import *
import os

class ReadData:
    def __init__(self, datatype, seq_len, pre_len, pre_sens_num, demo):
        self.datatype = datatype
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.pre_sens_num = pre_sens_num
        self.demo = demo
        self._read_data()

    def _read_data(self):
        if self.datatype == "freeway":
            print("loading freeway data...")
            data1 = load_csv(r"data/freeway/10105110", 8, "freeway")
            data2 = load_csv(r"data/freeway/10105310", 8, "freeway")
            data3 = load_csv(r"data/freeway/10105510", 8, "freeway")
            data4 = load_csv(r"data/freeway/10108210", 8, "freeway")
            data5 = load_csv(r"data/freeway/10106510", 8, "freeway")
            data6 = load_csv(r"data/freeway/1095110", 8, "freeway")
            data7 = load_csv(r"data/freeway/1095510", 8, "freeway")

        elif self.datatype == "urban":
            print("loading urban data...")
            data1 = load_csv(r"data/urban/401190", 5, "urban")
            data2 = load_csv(r"data/urban/401144", 7, "urban")
            data3 = load_csv(r"data/urban/401413", 11, "urban")
            data4 = load_csv(r"data/urban/401911", 8, "urban")
            data5 = load_csv(r"data/urban/401610", 10, "urban")
            data6 = load_csv(r"data/urban/401273", 8, "urban")
            data7 = load_csv(r"data/urban/401137", 8, "urban")

        # prepare train and test

        (
            train_data,
            train_w,
            train_d,
            self.label,
            test_data,
            test_w,
            test_d,
            self.test_l,
            self.test_med,
            self.test_min,
        ) = generate_data(
            data1,
            data2,
            data3,
            data4,
            data5,
            data6,
            data7,
            self.seq_len,
            self.pre_len,
            self.pre_sens_num,
        )

        self.train_data = np.reshape(
            train_data,
            (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1),
        )
        self.train_w = np.reshape(train_w, (train_w.shape[0], train_w.shape[1], 1))
        self.train_d = np.reshape(train_d, (train_d.shape[0], train_d.shape[1], 1))

        self.test_data = np.reshape(
            test_data, (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)
        )
        self.test_d = np.reshape(test_d, (test_d.shape[0], test_d.shape[1], 1))
        self.test_w = np.reshape(test_w, (test_w.shape[0], test_w.shape[1], 1))

        if self.demo:
            self.train_data = self.train_data[:3000]
            self.train_w = self.train_w[:3000]
            self.train_d = self.train_d[:3000]
            self.label = self.label[:3000]

            self.test_data = self.test_data[:500]
            self.test_w = self.test_w[:500]
            self.test_d = self.test_d[:500]
            self.test_l = self.test_l[:500]

        print("train_data shape:", self.train_data.shape)
        print("train_w shape:", self.train_w.shape)
        print("train_d shape:", self.train_d.shape)
        print("label shape:", self.label.shape)

        print("test_data shape:", self.test_data.shape)
        print("test_w shape:", self.test_w.shape)
        print("test_d shape:", self.test_d.shape)
        print("test_l shape:", self.test_l.shape)
        # print("test_med:", self.test_med)
        # print("test_min:", self.test_min)
