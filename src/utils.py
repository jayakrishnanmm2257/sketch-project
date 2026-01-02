import torch
import json
import os

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't want to update the learning rate based on the checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_vocab(vocab, path):
    with open(path, 'w') as f:
        json.dump(vocab, f)

def load_vocab(path):
    with open(path, 'r') as f:
        return json.load(f)

def text_to_labels(text, vocab, device):
    """
    Converts a comma-separated string to a multi-hot tensor.
    """
    vocab_size = len(vocab)
    label_vector = torch.zeros(1, vocab_size).to(device)
    
    attributes = [attr.strip() for attr in text.split(',')]
    for attr in attributes:
        if attr in vocab:
            label_vector[0, vocab[attr]] = 1
        else:
            print(f"Warning: Attribute '{attr}' not in vocabulary.")
            
    return label_vector
