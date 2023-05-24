from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class COCODataset(Dataset):

    def __init__(self, cocoid, caption, filepath, filename, sentid):
        self.cocoid = cocoid
        self.caption = caption
        # self.filepath = filepath
        self.filename = filename
        self.sentid = sentid
    
    def __len__(self):
        return len(self.cocoid)
    
    def __getitem__(self, item):
        
        cocoid = self.cocoid[item]
        # filepath = self.filepath[item]
        filename = self.filename[item]
        caption = self.caption[item]['raw']  #get the sentence which is stored in the form of dictionary raw: " "
        sentid = self.sentid[item]['sentid']
        img_fpath = os.path.join("http://images.cocodataset.org/val2014", filename)

        return {
            'cocoid': cocoid,
            'img_fpath': img_fpath,
            'caption': caption,
            'sentid': sentid
        }

def create_data_loader(df, batch_size):

    ds = COCODataset(
        cocoid = df['cocoid'],
        caption = df['sentences'],
        filepath = df['filepath'],
        filename = df['filename'],
        sentid = df['sentences']
    )

    #fix me ! batch_size

    return DataLoader(
        ds, 
        batch_size = batch_size,
        num_workers = 2
    )
