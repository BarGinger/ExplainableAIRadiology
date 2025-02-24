import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import zipfile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset import CheXpertDataset


# Global helper functions
def get_class_names():
    return [
        'Pleural Effusion'
    ]

def get_policies():
    return [
        'ones',
        'zeroes',
        'mixed'
    ]

# Global preprocessing functions

def prepare_dataset(dataframe, policy, class_names):
    """
    Prepare the dataset by filtering, shuffling, and filling missing values.

    Parameters:
    - dataframe: The input DataFrame containing the dataset.
    - policy: The policy to handle uncertain labels (-1). Options are "ones", "zeroes", "mixed".
    - class_names: List of class names for the medical conditions.

    Returns:
    - x_path: Numpy array of image paths.
    - y: Numpy array of labels corresponding to the class names.
    """
    # Filter the dataset to include only frontal images
    dataset_df = dataframe[dataframe['Frontal/Lateral'] == 'Frontal']
    
    # Shuffle the dataset
    df = dataset_df.sample(frac=1., random_state=1)
    
    # Fill missing values with zeros
    df.fillna(0, inplace=True)
    
    # Extract image paths and labels
    x_path = df["Path"].to_numpy()
    y_df = df[class_names]
    
    # Define classes to be treated as ones in the "mixed" policy
    class_ones = ['Atelectasis', 'Cardiomegaly']
    
    # Initialize the labels array
    y = np.empty(y_df.shape, dtype=int)
    
    # Define a dictionary to map policies to their corresponding actions
    policy_actions = {
        "ones": lambda cls: 1,
        "zeroes": lambda cls: 0,
        "mixed": lambda cls: 1 if cls in class_ones else 0
    }
    
    # Iterate over each row in the labels DataFrame
    for i, (index, row) in enumerate(y_df.iterrows()):
        labels = []
        # Iterate over each class name
        for cls in class_names:
            curr_val = row[cls]
            if curr_val:
                curr_val = float(curr_val)
                if curr_val == 1:
                    feat_val = 1
                elif curr_val == -1:
                    feat_val = policy_actions.get(policy, lambda cls: 0)(cls)
                else:
                    feat_val = 0
            else:
                feat_val = 0
            
            labels.append(feat_val)
        
        # Assign the labels to the corresponding row in the labels array
        y[i] = labels
    
    return x_path, y

def split_train_val(train_df, policy, class_names, test_size=0.2, random_state=42):
    """
    Split the training data into training and validation sets.

    Parameters:
    - train_df: DataFrame containing the training data.
    - policy: The policy to handle uncertain labels (-1). Options are "ones", "zeroes", "mixed".
    - class_names: List of class names for the medical conditions.
    - test_size: Proportion of the training data to include in the validation set.
    - random_state: Random seed for reproducibility.

    Returns:
    - train_df: DataFrame containing the training data.
    - val_df: DataFrame containing the validation data.
    """
    # Prepare the training dataset
    train_paths, train_labels = prepare_dataset(train_df, policy, class_names)
    
    # Split the training dataset into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=test_size, random_state=random_state
    )
    
    # Create DataFrames for the training and validation sets
    train_df = pd.DataFrame({'path': train_paths})
    train_labels_df = pd.DataFrame(train_labels, columns=class_names)
    train_df = pd.concat([train_df, train_labels_df], axis=1)
    
    val_df = pd.DataFrame({'path': val_paths})
    val_labels_df = pd.DataFrame(val_labels, columns=class_names)
    val_df = pd.concat([val_df, val_labels_df], axis=1)
    
    return train_df, val_df

def prepare_test_dataset(valid_df, policy, class_names):
    """
    Prepare the test dataset (original validation set).

    Parameters:
    - valid_df: DataFrame containing the original validation data.
    - policy: The policy to handle uncertain labels (-1). Options are "ones", "zeroes", "mixed".
    - class_names: List of class names for the medical conditions.

    Returns:
    - test_df: DataFrame containing the test data.
    """
    # Prepare the test dataset
    test_paths, test_labels = prepare_dataset(valid_df, policy, class_names)
    
    # Create DataFrame for the test set
    test_df = pd.DataFrame({'path': test_paths})
    test_labels_df = pd.DataFrame(test_labels, columns=class_names)
    test_df = pd.concat([test_df, test_labels_df], axis=1)
    
    return test_df


def get_datasets(zip_path='chexpert.zip'):
    """
    Get the training, validation, and test datasets.
    """
    # Read the training and validation data from the zip file
    original_train_df, test_df = read_zip(zip_path=zip_path)

    policies = get_policies()
    class_names = get_class_names()

    # Select the policy to handle uncertain labels (-1)
    # We started with mixed policy which changed -1  to 0 but got bad results so switched to ones policy
    # which changes -1 to 1
    selected_policy = policies[0]

    # Split the original training data into separate training 
    # and validation sets while preserving the original 
    # validation test set as the final test set.
    train_df, validation_df = split_train_val(original_train_df, selected_policy, class_names)
    
    # Prepare the test dataset
    test_df = prepare_test_dataset(test_df, selected_policy, class_names)

    return train_df, validation_df, test_df


def read_zip(zip_path='chexpert.zip'):
    """
    Reading training and validation data from a zip file.
    """
    original_train_df, test_df = None, None

    # Read CSV files from the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open('train.csv') as train_file:
            original_train_df = pd.read_csv(train_file)
        with zip_ref.open('valid.csv') as valid_file:
            test_df = pd.read_csv(valid_file)
    
    return original_train_df, test_df


def get_transform(augment=False):
    # Define the transformation pipeline for the images
    transform_list = [
        transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
    ]
    
    if augment:
        # Add data augmentation transformations to help with generalization
        # RandomHorizontalFlip: Randomly flip the image horizontally
        # RandomRotation: Randomly rotate the image by up to 10 degrees
        # RandomResizedCrop: Randomly crop the image to 224x224 pixels
        # ColorJitter: Randomly change the brightness, contrast, saturation, and hue of the image
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        ]
        transform_list = augmentations + transform_list
    
    transform = transforms.Compose(transform_list)
    return transform

def transform_dataset(df, zip_path='chexpert.zip', batch_size=16, shuffle=True, augment=False):
    """
    Transform the dataset into DataLoader objects for given dataframe.

    Parameters:
    - df: DataFrame containing the dataset.
    - zip_path: Path to the zip file containing the images.
    - batch_size: Number of samples in each batch.
    - shuffle: Whether to shuffle the data.
    - augment: Whether to apply data augmentation.

    Returns:
    - dataset: CheXpertDataset object containing the dataset.
    - loader: DataLoader object containing the dataset.
    - images: Batch of images from the DataLoader.
    - labels: Batch of labels from the DataLoader.
    """
    # Define the class names for the medical conditions
    class_names = get_class_names()

    transformer = get_transform(augment=augment)

    # Create the training dataset with the defined transformations 
    dataset = CheXpertDataset(dataframe=df, class_names=class_names, zip_path=zip_path, transform=transformer)

    # Create DataLoader for the training dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Verify data loading by fetching a batch of images and labels from the training DataLoader
    images, labels = next(iter(loader))

    return dataset, loader, images, labels
