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

class Fakeddit(DataLoader):

    def __init__(self, fakeid, clean_title, image_url, way_labels):
        self.fakeid = fakeid
        self.clean_title = clean_title
        self.image_url = image_url
        self.way_labels = way_labels

    def __len__(self):
        return len(self.fakeid)

    def __getitem__(self, item):

        fakeid = self.fakeid[item]
        clean_title = self.clean_title[item]
        image_url = self.image_url[item]
        way_labels = self.way_labels[item]

        return {
            'fakeid': fakeid,
            'clean_title': clean_title,
            'image_url': image_url,
            'way_labels': way_labels
        }

def create_fakedata_loader(df, batch_size, label):

    # print(label)
    # print(type(df))
    # print(df.columns)
    # print(df.shape)
    # print(df.dtypes)
    # ds_fil = df[df['6_way_label'] == label]
    # ds_fil = ds_fil.reset_index(drop=True)

    # # print(ds_fil['6_way_label'])
    # print(type(ds_fil))
    # print(ds_fil.shape)
    # print(ds_fil.dtypes)

    # print(ds_fil.columns)
    # print(df['6_way_label'].value_counts())
    # exit()
    # df['6_way_label'] = df['6_way_label'].astype(int)
    
    df = df.dropna(subset=['image_url', 'clean_title', 'created_utc', '6_way_label'])
    df = df.reset_index(drop=True)

    filtered_df = df[df['6_way_label'] == label].reset_index(drop=True)

    print(f"for label, {label}")
    print(filtered_df.shape)
    # Create the Fakeddit dataset using the filtered DataFrame
    ds = Fakeddit(
        fakeid=filtered_df['created_utc'],
        clean_title=filtered_df['clean_title'],
        image_url=filtered_df['image_url'],
        way_labels=filtered_df['6_way_label']
    )

    # print(df.shape)

    # ds = Fakeddit(
    #     fakeid = df['id'],
    #     clean_title = df['clean_title'],
    #     image_url = df['image_url'],
    #     way_labels = df['6_way_label']
    # )


    return DataLoader(
        ds, 
        batch_size = batch_size,
        num_workers = 2
    )


