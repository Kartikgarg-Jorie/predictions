import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        enc = self.tokenizer(
            text,
            truncation=True,       # max_length Accepted by the model
            padding="max_length",  # to the max length of sequence
            max_length=self.max_length,
            return_tensors="True", # model data format in pytorch_tensor
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label,
        }

class CustomTrainer(Trainer):

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

class CodeTrainer:
    def __init__(
        self,
        df,
        label_column,
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        output_dir="model_out",
        type = "multi_label_classification",
    ):
        self.df = df
        self.label_column = label_column
        self.model_name = model_name
        self.output_dir = output_dir

        # Load tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

        # MultiLabel Binarizer
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(df[label_column])

        label_counts = np.sum(self.mlb.transform(df[label_column]), axis=0)
        neg_counts = len(df) - label_counts

        pos_weight = neg_counts / (label_counts + 1e-6)
        self.pos_weight = torch.tensor(pos_weight, dtype=torch.float)

        # Split
        train_df, val_df = train_test_split(
            df, test_size=0.15, random_state=42)

        self.train_dataset = MultiLabelDataset(
            train_df["TEXT_CLEAN"].tolist(),
            self.mlb.transform(train_df[label_column]),
            self.tokenizer,
        )

        self.val_dataset = MultiLabelDataset(
            val_df["TEXT_CLEAN"].tolist(),
            self.mlb.transform(val_df[label_column]),
            self.tokenizer,
        )

        # Load base model
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.mlb.classes_),
            problem_type= type,
        )

    # Train model function

    def train_model(self):
        os.makedirs(self.output_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3,
            learning_rate=2e-5,
            num_train_epochs=4,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = CustomTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset
            
        )

        trainer.train()
        trainer.save_model(self.output_dir)
        
        # Save tokenizer inside model folder
        self.tokenizer.save_pretrained(self.output_dir)

        # Save class labels
        np.save(
            os.path.join(self.output_dir, "mlb_classes.npy"),
            np.array(self.mlb.classes_),
        )

if __name__ == "__main__":
    Dateset_FILE = "dataset/module1105/1105 (2).csv"

    df = pd.read_csv(Dateset_FILE, dtype=str)

    df = df.apply(lambda col: col.str.strip() 
                  if col.dtype == "object" else col)


    df = df.dropna(how="all")             # remove blank rows
    df = df.dropna(axis=1, how="all")     # remove blank columns

    icd_cols = [c for c in df.columns if c.upper().startswith("ICD")]
    cpt_cols = [c for c in df.columns if c.upper().startswith("CPT")]

    def clean_code_list(code_list):
        cleaned = []
        for c in code_list:
            if pd.isna(c):
                continue
            c = str(c).strip()        # remove outer whitespace
            c = c.replace(" ", "")    # remove inner spaces
            if c != "":
                cleaned.append(c)
        return cleaned

    df["ICD_LIST"] = df[icd_cols].apply(
        lambda row: clean_code_list(row.values), axis=1
    )

    df["CPT_LIST"] = df[cpt_cols].apply(
        lambda row: clean_code_list(row.values), axis=1
    )

    # Remove rows with no ICD/CPT codes at all
    df = df[df["ICD_LIST"].apply(len) > 0]
    df = df[df["CPT_LIST"].apply(len) > 0]


    df["TEXT_CLEAN"] = df["Text"].fillna("").astype(str).str.strip()
    
    icd_trainer = CodeTrainer(
        df, label_column="ICD_LIST", output_dir="ICD_MODEL")
    icd_trainer.train_model()
   
    cpt_trainer = CodeTrainer(
        df, label_column="CPT_LIST", output_dir="CPT_MODEL")
    cpt_trainer.train_model()
    

# class Upload():
#     def upload(self):
#         self.icd_trainer = CodeTrainer(df, label_column="ICD_LIST", output_dir="ICD_MODEL")
#         self.cpt_trainer = CodeTrainer(df, label_column="CPT_LIST", output_dir="CPT_MODEL")
#         self.icd_trainer.train_model()
#         self.cpt_trainer.train_model()
      

