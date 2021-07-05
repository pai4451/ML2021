import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
import random
import math
from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append('./Conformer')
from CF import Conformer
import csv

class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len
 
        # Load the mapping from speaker neme to their corresponding id. 
        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping["speaker2id"]
 
        # Load metadata of training data.
        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(open(metadata_path))["speakers"]
 
        # Get the total number of speaker.
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))
 
        # Segmemt mel-spectrogram into "segment_len" frames.
        if len(mel) > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker
 
    def get_speaker_number(self):
        return self.speaker_num




def collate_batch(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad -20 which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
    """Generate dataloader"""
    dataset = myDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    # Split dataset into training dataset and validation dataset
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    return train_loader, valid_loader, speaker_num



class Classifier(nn.Module):
    def __init__(self, config, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        #   https://github.com/Masao-Someki/Conformer
        self.encoder_layer = Conformer(**config)
        #self.encoder_layer = nn.TransformerEncoderLayer(
        #  d_model=d_model, dim_feedforward=256, nhead=1
        #)
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
          #nn.Linear(d_model, d_model),
          #nn.ReLU(),
          nn.Linear(d_model, n_spks),
        )


    def forward(self, mels):
        """
        args:
          mels: (batch size, length, 40)
        return:
          out: (batch size, n_spks)
        """
        out = self.prenet(mels)
        # out: (batch size, length, d_model)
        #out = out.permute(1, 0, 2)
        # out: (length, batch size, d_model)
        # For transformer: The encoder layer expect features in the shape of (length, batch size, d_model).
        # For conformer: The encoder layer expect features in the shape of (batch size, length, d_model).
        out = self.encoder_layer(out)
        # out: (length, batch size, d_model)
        # conformer out: (batch size, length, d_model)
        #out = out.transpose(0, 1)
        # out: (batch size, length, d_model)
        # mean pooling
        stats = out.mean(dim=1)
        # stats: (batch size, d_model)
        out = self.pred_layer(stats)
        # out: (batch, n_spks)
        return out







def get_cosine_schedule_with_warmup(
      optimizer: Optimizer,
      num_warmup_steps: int,
      num_training_steps: int,
      num_cycles: float = 0.5,
      last_epoch: int = -1,
    ):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

    Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
          max(1, num_training_steps - num_warmup_steps)
        )
        return max(
          0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    mels, labels = batch # size [batch size, length, 40] , [batch size]
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)
    # outs: (batch, n_spks)

    loss = criterion(outs, labels)

    # Get the speaker id with highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy




def valid(dataloader, model, criterion, device):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
          loss=f"{running_loss / (i+1):.2f}",
          accuracy=f"{running_accuracy / (i+1):.2f}",
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)





def train_parse_args():

    """arguments"""

    config = {
        "data_dir": "./Dataset",
        "batch_size": 32,
        "n_workers": 8,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 10000,
        "total_steps": 200000,
        "model_config":{
            "config1":{
                "d_model":80,
                "ff1_hsize": 2048,
                "ff1_dropout": 0.2,
                "n_head": 2,
                "mha_dropout": 0.2,
                "kernel_size": 3,
                "conv_dropout": 0.2,
                "ff2_hsize": 2048,
                "ff2_dropout": 0.2
            },
            "config2":{
                "d_model":80,
                "ff1_hsize": 2048,
                "ff1_dropout": 0.2,
                "n_head": 4,
                "mha_dropout": 0.2,
                "kernel_size": 3,
                "conv_dropout": 0.2,
                "ff2_hsize": 2048,
                "ff2_dropout": 0.2
            },
            "config3":{
                "d_model":80,
                "ff1_hsize": 4096,
                "ff1_dropout": 0.2,
                "n_head": 2,
                "mha_dropout": 0.2,
                "kernel_size": 3,
                "conv_dropout": 0.2,
                "ff2_hsize": 4096,
                "ff2_dropout": 0.2
            },
        },
        "model_path": {
            "config1":"./model96214.ckpt",
            "config2":"./model96142.ckpt",
            "config3":"./model96642.ckpt"},
    }

    return config

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_main(data_dir,batch_size,n_workers,valid_steps,warmup_steps,total_steps,save_steps,
                model_config, model_path):
    
    """Main function."""
    
    for i in model_config:
    
        same_seeds(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Info]: Use {device} now!")

        train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
        train_iterator = iter(train_loader)
        print(f"[Info]: Finish loading data!",flush = True)

        model = Classifier(model_config[i],n_spks=speaker_num).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        print(f"[Info]: Finish creating model!",flush = True)

        best_accuracy = -1.0
        best_state_dict = None

        pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        for step in range(total_steps):
            # Get data
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            loss, accuracy = model_fn(batch, model, criterion, device)
            batch_loss = loss.item()
            batch_accuracy = accuracy.item()

            # Updata model
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Log
            pbar.update()
            pbar.set_postfix(
              loss=f"{batch_loss:.2f}",
              accuracy=f"{batch_accuracy:.2f}",
              step=step + 1,
            )

            # Do validation
            if (step + 1) % valid_steps == 0:
                pbar.close()

                valid_accuracy = valid(valid_loader, model, criterion, device)

                # keep the best model
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    best_state_dict = model.state_dict()

                pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

            # Save the best model so far.
            if (step + 1) % save_steps == 0 and best_state_dict is not None:
                torch.save(best_state_dict, model_path[i])
                pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

        pbar.close()



class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / "testdata.json"
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata["utterances"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance["feature_path"]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        return feat_path, mel


def inference_collate_batch(batch):
    """Collate a batch of data."""
    feat_paths, mels = zip(*batch)

    return feat_paths, torch.stack(mels)



def test_main(data_dir,model_path,output_path,model_config):

    """Main function."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())

    dataset = InferenceDataset(data_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=inference_collate_batch,
    )
    print(f"[Info]: Finish loading data!",flush = True)

    speaker_num = len(mapping["id2speaker"])
    model1 = Classifier(model_config["config1"], n_spks=speaker_num).to(device)
    model1.load_state_dict(torch.load(model_path['model1']))
    model1.eval()
    model2 = Classifier(model_config["config2"], n_spks=speaker_num).to(device)
    model2.load_state_dict(torch.load(model_path['model2']))
    model2.eval()
    model3 = Classifier(model_config["config3"], n_spks=speaker_num).to(device)
    model3.load_state_dict(torch.load(model_path['model3']))
    model3.eval()
    print(f"[Info]: Finish creating model!",flush = True)

    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs1 = model1(mels)
            outs2 = model2(mels)
            outs3 = model3(mels)
            outs = (outs1+outs2+outs3) / 3
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


def test_parse_args():

    """arguments"""
    config = {
        "data_dir": "./Dataset",
        "model_path": {
            "model1":"./model96214.ckpt",
            "model2":"./model96142.ckpt",
            "model3":"./model96642.ckpt"},
        "output_path": "./fusion.csv",
        "model_config":{
            "config1":{
                "d_model":80,
                "ff1_hsize": 2048,
                "ff1_dropout": 0.2,
                "n_head": 2,
                "mha_dropout": 0.2,
                "kernel_size": 3,
                "conv_dropout": 0.2,
                "ff2_hsize": 2048,
                "ff2_dropout": 0.2
            },
            "config2":{
                "d_model":80,
                "ff1_hsize": 2048,
                "ff1_dropout": 0.2,
                "n_head": 4,
                "mha_dropout": 0.2,
                "kernel_size": 3,
                "conv_dropout": 0.2,
                "ff2_hsize": 2048,
                "ff2_dropout": 0.2
            },
            "config3":{
                "d_model":80,
                "ff1_hsize": 4096,
                "ff1_dropout": 0.2,
                "n_head": 2,
                "mha_dropout": 0.2,
                "kernel_size": 3,
                "conv_dropout": 0.2,
                "ff2_hsize": 4096,
                "ff2_dropout": 0.2
            },
        }
    }

    return config


if __name__ == "__main__":

    train_main(**train_parse_args())
    test_main(**test_parse_args())



# References
# This code is modified from TA's sample code in NTU machine learning course
# Conformer source I used: https://github.com/Masao-Someki/Conformer