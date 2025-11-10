#!/usr/bin/env python
# coding=utf-8

import os, sys
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
import numpy as np
from .data_augmentation import augment_sequence

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor

class KTDataset(Dataset):
    """Dataset for KT
        can use to init dataset for: (for models except dkt_forget)
            train data, valid data
            common test data(concept level evaluation), real educational scenario test data(question level evaluation).
    Args:
        file_path (str): train_valid/test file path
        input_type (list[str]): the input type of the dataset, values are in ["questions", "concepts"]
        folds (set(int)): the folds used to generate dataset, -1 for test data
        qtest (bool, optional): is question evaluation or not. Defaults to False.
    """
    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            pd.to_pickle(save_data, processed_data)
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                self.dori, self.dqtest = pd.read_pickle(processed_data)
            else:
                self.dori = pd.read_pickle(processed_data)
                for key in self.dori:
                    self.dori[key] = self.dori[key]#[:100]
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """return the dataset length
        Returns:
            int: the length of the dataset
        """
        return len(self.dori["rseqs"])

    def __init__(self, file_path, input_type, folds, qtest=False):
        super(KTDataset, self).__init__()
        sequence_path = file_path
        self.input_type = input_type
        self.qtest = qtest
        # 对 folds 进行排序并转换为字符串，方便后续生成文件名
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        # 根据是否进行问题测试生成不同的文件名
        if self.qtest:
            processed_data = file_path + folds_str + "_qtest.pkl"
        else:
            processed_data = file_path + folds_str + ".pkl"

        # 检查处理后的数据文件是否存在
        if not os.path.exists(processed_data):
            print(f"Start preprocessing {file_path} fold: {folds_str}...")
            if self.qtest:
                self.dori, self.dqtest = self.__load_data__(sequence_path, folds)
                save_data = [self.dori, self.dqtest]
            else:
                self.dori = self.__load_data__(sequence_path, folds)
                save_data = self.dori
            # 将处理后的数据保存为 pickle 文件
            try:
                pd.to_pickle(save_data, processed_data)
            except Exception as e:
                print(f"Error saving data to {processed_data}: {e}")
        else:
            print(f"Read data from processed file: {processed_data}")
            if self.qtest:
                try:
                    self.dori, self.dqtest = pd.read_pickle(processed_data)
                except Exception as e:
                    print(f"Error reading data from {processed_data}: {e}")
            else:
                try:
                    self.dori = pd.read_pickle(processed_data)
                    for key in self.dori:
                        self.dori[key] = self.dori[key]  # 此处注释掉的切片操作可以根据实际需求决定是否保留
                except Exception as e:
                    print(f"Error reading data from {processed_data}: {e}")
        # 打印文件路径和数据集长度信息
        print(f"file path: {file_path}, qlen: {len(self.dori['qseqs'])}, clen: {len(self.dori['cseqs'])}, rlen: {len(self.dori['rseqs'])}")

    def __len__(self):
        """
        返回数据集的长度
        Returns:
            int: 数据集的长度
        """
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        """
        根据索引获取数据
        Args:
            index (int): 要获取的数据的索引
        Returns:
            (tuple): 包含以下内容的元组:
                - **q_seqs (torch.tensor)**: 0 到 seqlen - 2 交互的问题 ID 序列
                - **c_seqs (torch.tensor)**: 0 到 seqlen - 2 交互的知识概念 ID 序列
                - **r_seqs (torch.tensor)**: 0 到 seqlen - 2 交互的响应 ID 序列
                - **qshft_seqs (torch.tensor)**: 1 到 seqlen - 1 交互的问题 ID 序列
                - **cshft_seqs (torch.tensor)**: 1 到 seqlen - 1 交互的知识概念 ID 序列
                - **rshft_seqs (torch.tensor)**: 1 到 seqlen - 1 交互的响应 ID 序列
                - **mask_seqs (torch.tensor)**: 掩码值序列，形状为 seqlen - 1
                - **select_masks (torch.tensor)**: 是否选择计算性能，0 表示不选择，1 表示选择，仅适用于 1 到 seqlen - 1，形状为 seqlen - 1
                - **dcur (dict)**: 仅在 self.qtest 为 True 时使用，用于问题级评估
        """
        dcur = dict()
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            if len(self.dori[key]) == 0:
                dcur[key] = self.dori[key]
                dcur["shft_" + key] = self.dori[key]
                continue
            # 生成序列和移位序列
            seqs = self.dori[key][index][:-1] * mseqs
            shft_seqs = self.dori[key][index][1:] * mseqs
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]

        # 以 50% 的概率进行数据增强
        if random.random() < 0.5:
            q = dcur["qseqs"]
            c = dcur["cseqs"]
            r = dcur["rseqs"]
            try:
                q, c, r = augment_sequence(q, c, r)
            except Exception as e:
                print(f"Error in data augmentation: {e}")
            dcur["qseqs"] = q
            dcur["cseqs"] = c
            dcur["rseqs"] = r

            qshft = dcur["shft_qseqs"]
            cshft = dcur["shft_cseqs"]
            rshft = dcur["shft_rseqs"]
            try:
                qshft, cshft, rshft = augment_sequence(qshft, cshft, rshft)
            except Exception as e:
                print(f"Error in data augmentation: {e}")
            dcur["shft_qseqs"] = qshft
            dcur["shft_cseqs"] = cshft
            dcur["shft_rseqs"] = rshft

        if not self.qtest:
            return dcur
        else:
            dqtest = dict()
            for key in self.dqtest:
                dqtest[key] = self.dqtest[key][index]
            return dcur, dqtest

    def __load_data__(self, sequence_path, folds, pad_val=-1):
        """
        加载数据
        Args:
            sequence_path (str): 序列文件的路径
            folds (list[int]): 折叠列表
            pad_val (int, optional): 填充值。默认为 -1。
        Returns:
            (tuple): 包含以下内容的元组:
                - **q_seqs (torch.tensor)**: 0 到 seqlen - 1 交互的问题 ID 序列
                - **c_seqs (torch.tensor)**: 0 到 seqlen - 1 交互的知识概念 ID 序列
                - **r_seqs (torch.tensor)**: 0 到 seqlen - 1 交互的响应 ID 序列
                - **mask_seqs (torch.tensor)**: 掩码值序列，形状为 seqlen - 1
                - **select_masks (torch.tensor)**: 是否选择计算性能，0 表示不选择，1 表示选择，仅适用于 1 到 seqlen - 1，形状为 seqlen - 1
                - **dqtest (dict)**: 仅在 self.qtest 为 True 时不为空，用于问题级评估
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "tseqs": [], "utseqs": [], "smasks": []}
        # 读取 CSV 文件
        try:
            df = pd.read_csv(sequence_path)  # [0:1000] 注释掉的切片操作可以根据实际需求决定是否保留
            df = df[df["fold"].isin(folds)]
        except Exception as e:
            print(f"Error reading CSV file {sequence_path}: {e}")
            return None

        interaction_num = 0
        dqtest = {"qidxs": [], "rests": [], "orirow": []}
        for i, row in df.iterrows():
            # 根据输入类型加载数据
            if "concepts" in self.input_type:
                dori["cseqs"].append([int(_) for _ in row["concepts"].split(",")])
            if "questions" in self.input_type:
                dori["qseqs"].append([int(_) for _ in row["questions"].split(",")])
            if "timestamps" in row:
                dori["tseqs"].append([int(_) for _ in row["timestamps"].split(",")])
            if "usetimes" in row:
                dori["utseqs"].append([int(_) for _ in row["usetimes"].split(",")])

            dori["rseqs"].append([int(_) for _ in row["responses"].split(",")])
            dori["smasks"].append([int(_) for _ in row["selectmasks"].split(",")])

            interaction_num += dori["smasks"][-1].count(1)

            if self.qtest:
                dqtest["qidxs"].append([int(_) for _ in row["qidxs"].split(",")])
                dqtest["rests"].append([int(_) for _ in row["rest"].split(",")])
                dqtest["orirow"].append([int(_) for _ in row["orirow"].split(",")])

        # 将数据转换为张量
        for key in dori:
            if key not in ["rseqs"]:  # in ["smasks", "tseqs"]:
                dori[key] = LongTensor(dori[key])
            else:
                dori[key] = FloatTensor(dori[key])

        # 生成掩码序列
        mask_seqs = (dori["cseqs"][:, :-1] != pad_val) * (dori["cseqs"][:, 1:] != pad_val)
        dori["masks"] = mask_seqs

        dori["smasks"] = (dori["smasks"][:, 1:] != pad_val)
        print(f"interaction_num: {interaction_num}")
        # print("load data tseqs: ", dori["tseqs"])

        if self.qtest:
            for key in dqtest:
                dqtest[key] = LongTensor(dqtest[key])[:, 1:]
            return dori, dqtest
        return dori
    
#  a = 0 b = 1


def change_random_A_to_B(lst, shft_lst, a,p):

    for i in range(1, len(lst)):
        #num = np.random.randint(0,len(lst))
        prob = np.random.rand()
        if prob < p:
            if lst[i] == a:
                lst[i] = 1 - lst[i]
                shft_lst[i-1] = 1 - shft_lst[i-1]
                break
        #if lst[num] == a:
            #lst[num] = b
            #shft_lst[num+1] = b
            #break
        
    return lst, shft_lst