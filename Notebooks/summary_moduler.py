# System Library
import os

# Data Wrangling Libraries
import pandas as pd
import numpy as np
import json
import gc
import textwrap
from termcolor import colored

# Machine Learning Libraries
import torch 
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer
from tqdm.auto import tqdm


# Graph Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from matplotlib import rc


MODEL_NAME = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
EPOCHS = 3
BATCH_SIZE = 8

class SummaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame = None,
        tokenizer: T5Tokenizer = None,
        title_max_token_len: int = 512,
        content_max_token_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.title_max_token_len = title_max_token_len
        self.content_max_token_len = content_max_token_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        title = data_row['title']

        title_encoding = self.tokenizer(
            title,
            max_length = self.title_max_token_len,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = 'pt'
        )

        content = data_row['content']

        content_encoding = self.tokenizer(
            content,
            max_length = self.content_max_token_len,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            add_special_tokens = True,
            return_tensors = 'pt'
        )

        labels = content_encoding['input_ids']
        labels[labels == 0] = -100

        return dict(
            title = title,
            content = content,
            title_input_ids = title_encoding['input_ids'].flatten(),
            title_attention_mask = title_encoding['attention_mask'].flatten(),
            labels = labels.flatten(),
            labels_attention_mask = content_encoding['attention_mask'].flatten()
        )

class SummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 8,
        title_max_token_len: int = 512,
        content_max_token_len: int = 128
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.title_max_token_len = title_max_token_len
        self.content_max_token_len = content_max_token_len
        self.train_df = train_df
        self.test_df = test_df

    def setup(self, stage=None):

        self.train_dataset = SummaryDataset(
            self.train_df,
            self.tokenizer,
            self.title_max_token_len,
            self.content_max_token_len
        )

        self.test_dataset = SummaryDataset(
            self.test_df,
            self.tokenizer,
            self.title_max_token_len,
            self.content_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def validation_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

class SummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        input_ids = batch['title_input_ids']
        attention_mask = batch['title_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_size):
        input_ids = batch['title_input_ids']
        attention_mask = batch['title_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("validation_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_size):
        input_ids = batch['title_input_ids']
        attention_mask = batch['title_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.001)
