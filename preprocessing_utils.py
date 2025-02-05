import pandas as pd
import numpy as np

def prepare_dataset(dataframe,policy,class_names):
    dataset_df = dataframe[dataframe['Frontal/Lateral'] == 'Frontal'] #take frontal pics only
    df = dataset_df.sample(frac=1., random_state=1)
    df.fillna(0, inplace=True) #fill the with zeros
    x_path, y_df = df["Path"].to_numpy(), df[class_names]
    class_ones = ['Atelectasis', 'Cardiomegaly']
    y = np.empty(y_df.shape, dtype=int)
    for i, (index, row) in enumerate(y_df.iterrows()):
        labels = []
        for cls in class_names:
            curr_val = row[cls]
            feat_val = 0
            if curr_val:
                curr_val = float(curr_val)
                if curr_val == 1:
                    feat_val = 1
                elif curr_val == -1:
                    if policy == "ones":
                        feat_val = 1
                    elif policy == "zeroes":
                        feat_val = 0
                    elif policy == "mixed":
                        if cls in class_ones:
                            feat_val = 1
                        else:
                            feat_val = 0
                    else:
                        feat_val = 0
                else:
                    feat_val = 0
            else:
                feat_val = 0
            
            labels.append(feat_val)
            
        y[i] = labels
        
    return x_path,y