
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
import math
import random





# Configuration
class Config:
    DATA_PATH = "/content/drive/MyDrive/TensorData/padded_dataset.pt"
    EMBEDDING_DIM = 512
    HIDDEN_SIZE = 512
    ENCODER_LAYERS = 2
    DECODER_LAYERS = 4
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_data():
    """Load and prepare the dataset"""
    print("📁 Loading dataset...")
    data = torch.load(Config.DATA_PATH)
    
    # Extract components
    src_tensor = data["src_tensor"]     # Urdu tokens (N, max_src_len)
    src_lengths = data["src_lengths"]
    tgt_tensor = data["tgt_tensor"]     # Roman tokens (N, max_tgt_len)
    tgt_lengths = data["tgt_lengths"]
    vocab_src = data["vocab_src"]       # Urdu vocabulary
    vocab_tgt = data["vocab_tgt"]       # Roman vocabulary
    
    # Create reverse mappings for decoding
    id2tgt = {i: token for token, i in vocab_tgt.items()}
    id2src = {i: token for token, i in vocab_src.items()}
    
    # Get padding indices
    PAD_SRC = vocab_src["<pad>"]
    PAD_TGT = vocab_tgt["<pad>"]
    
    print(f"✅ Data loaded:")
    print(f"   - Source vocabulary size: {len(vocab_src)}")
    print(f"   - Target vocabulary size: {len(vocab_tgt)}")
    print(f"   - Number of samples: {src_tensor.size(0)}")
    print(f"   - Max source length: {src_tensor.size(1)}")
    print(f"   - Max target length: {tgt_tensor.size(1)}")
    
    return {
        'src_tensor': src_tensor, 'src_lengths': src_lengths,
        'tgt_tensor': tgt_tensor, 'tgt_lengths': tgt_lengths,
        'vocab_src': vocab_src, 'vocab_tgt': vocab_tgt,
        'id2src': id2src, 'id2tgt': id2tgt,
        'PAD_SRC': PAD_SRC, 'PAD_TGT': PAD_TGT
    }


class TransliterationDataset(Dataset):
    def __init__(self, src_tensor, tgt_tensor, src_lengths, tgt_lengths):
        self.src_tensor = src_tensor
        self.tgt_tensor = tgt_tensor
        self.src_lengths = src_lengths
        self.tgt_lengths = tgt_lengths
    
    def __len__(self):
        return len(self.src_tensor)
    
    def __getitem__(self, idx):
        return {
            'src': self.src_tensor[idx],
            'tgt': self.tgt_tensor[idx],
            'src_len': self.src_lengths[idx],
            'tgt_len': self.tgt_lengths[idx]
        }

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, pad_idx):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM encoder (2 layers as specified)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Linear layer to project bidirectional hidden states to decoder hidden size
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_projection = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, src, src_lengths):
        batch_size = src.size(0)
        
        # Embedding
        embedded = self.dropout(self.embedding(src))
        
        # Pack padded sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # hidden: (num_layers * 2, batch, hidden_size) -> (num_layers, batch, hidden_size * 2)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)
        cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)
        
        # Project to decoder hidden size
        hidden = self.hidden_projection(hidden)
        cell = self.cell_projection(cell)
        
        return output, (hidden, cell)




class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, pad_idx):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM decoder (4 layers as specified)
        self.lstm = nn.LSTM(
            embedding_dim,  # Only embedding input
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, tgt, hidden, cell):
        # Embedding with dropout
        embedded = self.dropout(self.embedding(tgt))
        
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # Project to vocabulary
        output = self.out(output)
        
        return output, (hidden, cell)

class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, hidden_size, 
                 encoder_layers, decoder_layers, dropout, src_pad_idx, tgt_pad_idx):
        super(Seq2SeqModel, self).__init__()
        
        self.encoder = BiLSTMEncoder(
            src_vocab_size, embedding_dim, hidden_size, 
            encoder_layers, dropout, src_pad_idx
        )
        
        self.decoder = LSTMDecoder(
            tgt_vocab_size, embedding_dim, hidden_size,
            decoder_layers, dropout, tgt_pad_idx
        )
        
        # Initialize decoder hidden states from encoder
        self.init_hidden = nn.Linear(hidden_size, hidden_size)
        self.init_cell = nn.Linear(hidden_size, hidden_size)
        
        self.tgt_pad_idx = tgt_pad_idx
    
    def forward(self, src, tgt, src_lengths, tgt_lengths):
        # Encode
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(src, src_lengths)
        
        # Initialize decoder states from encoder's final hidden state
        decoder_hidden = torch.zeros(self.decoder.num_layers, src.size(0), 
                                   self.decoder.hidden_size, device=src.device)
        decoder_cell = torch.zeros(self.decoder.num_layers, src.size(0), 
                                 self.decoder.hidden_size, device=src.device)
        
        # Use encoder's final states for initialization of first layer
        decoder_hidden[0] = self.init_hidden(encoder_hidden[-1])
        decoder_cell[0] = self.init_cell(encoder_cell[-1])
        
        # Decode (teacher forcing during training)
        decoder_input = tgt[:, :-1]  # Exclude last token
        
        outputs, _ = self.decoder(decoder_input, decoder_hidden, decoder_cell)
        
        return outputs
    



def calculate_bleu(predictions, targets, id2tgt):
    """Calculate BLEU score"""
    bleu_scores = []
    smooth_func = SmoothingFunction().method1
    
    for pred, target in zip(predictions, targets):
        # Convert to tokens
        pred_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in pred]
        target_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in target]
        
        # Remove padding and special tokens
        pred_tokens = [t for t in pred_tokens if t not in ['<pad>', '<sos>', '<eos>']]
        target_tokens = [t for t in target_tokens if t not in ['<pad>', '<sos>', '<eos>']]
        
        if len(pred_tokens) > 0 and len(target_tokens) > 0:
            score = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smooth_func)
            bleu_scores.append(score)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0




def calculate_cer(predictions, targets, id2tgt):
    """Calculate Character Error Rate using Levenshtein distance"""
    distances = []
    
    for pred, target in zip(predictions, targets):
        # Convert to strings
        pred_str = ''.join([id2tgt.get(idx.item(), '') for idx in pred])
        target_str = ''.join([id2tgt.get(idx.item(), '') for idx in target])
        
        # Remove special tokens
        pred_str = pred_str.replace('<pad>', '').replace('<sos>', '').replace('<eos>', '')
        target_str = target_str.replace('<pad>', '').replace('<sos>', '').replace('<eos>', '')
        
        if len(target_str) > 0:
            distance = Levenshtein.distance(pred_str, target_str) / len(target_str)
            distances.append(distance)
    
    return np.mean(distances) if distances else 1.0

def calculate_perplexity(loss):
    """Calculate perplexity from cross-entropy loss"""
    return math.exp(loss)


def evaluate_model(model, dataloader, criterion, id2tgt, device):
    """Evaluate model and return metrics"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_len']
            tgt_lengths = batch['tgt_len']
            
            # Forward pass
            outputs = model(src, tgt, src_lengths, tgt_lengths)
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                           tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=-1)
            
            all_predictions.extend(predictions.cpu())
            all_targets.extend(tgt[:, 1:].cpu())
    
    avg_loss = total_loss / len(dataloader)
    perplexity = calculate_perplexity(avg_loss)
    bleu = calculate_bleu(all_predictions, all_targets, id2tgt)
    cer = calculate_cer(all_predictions, all_targets, id2tgt)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'bleu': bleu,
        'cer': cer
    }


def train_model(model, train_loader, val_loader, test_loader, optimizer, criterion, 
                epochs, device, id2tgt, freeze_encoder=True):
    """Train the model with option to freeze encoder"""
    
    # Freeze encoder parameters if specified
    if freeze_encoder:
        print("🔒 Freezing encoder parameters - only training decoder")
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        print("🔓 Training both encoder and decoder")
    
    train_losses = []
    val_losses = []
    
    print(f"\n🚀 Starting training on {device}")
    print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_len']
            tgt_lengths = batch['tgt_len']
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(src, tgt, src_lengths, tgt_lengths)
            
            # Calculate loss (ignore padding tokens)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                           tgt[:, 1:].reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, id2tgt, device)
        val_losses.append(val_metrics['loss'])
        
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Val Perplexity: {val_metrics['perplexity']:.4f}")
        print(f"   Val BLEU: {val_metrics['bleu']:.4f}")
        print(f"   Val CER: {val_metrics['cer']:.4f}")
        print("-" * 50)
    
    # Final evaluation on test set
    print("\n🧪 Final Test Evaluation:")
    test_metrics = evaluate_model(model, test_loader, criterion, id2tgt, device)
    print(f"   Test Loss: {test_metrics['loss']:.4f}")
    print(f"   Test Perplexity: {test_metrics['perplexity']:.4f}")
    print(f"   Test BLEU: {test_metrics['bleu']:.4f}")
    print(f"   Test CER: {test_metrics['cer']:.4f}")
    
    return train_losses, val_losses, test_metrics





def show_examples(model, test_loader, id2src, id2tgt, device, num_examples=5):
    """Show qualitative examples of translations"""
    model.eval()
    examples_shown = 0
    
    print("\n🔍 Translation Examples:")
    print("=" * 80)
    
    with torch.no_grad():
        for batch in test_loader:
            if examples_shown >= num_examples:
                break
                
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_len']
            tgt_lengths = batch['tgt_len']
            
            outputs = model(src, tgt, src_lengths, tgt_lengths)
            predictions = torch.argmax(outputs, dim=-1)
            
            for i in range(min(src.size(0), num_examples - examples_shown)):
                # Convert to readable text
                src_tokens = [id2src.get(idx.item(), '<unk>') for idx in src[i]]
                tgt_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in tgt[i, 1:]]  # Skip <sos>
                pred_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in predictions[i]]
                
                # Clean up tokens
                src_text = ' '.join([t for t in src_tokens if t != '<pad>']).replace('<pad>', '').strip()
                tgt_text = ' '.join([t for t in tgt_tokens if t not in ['<pad>', '<eos>']]).strip()
                pred_text = ' '.join([t for t in pred_tokens if t not in ['<pad>', '<eos>']]).strip()
                
                print(f"Example {examples_shown + 1}:")
                print(f"   Source (Urdu): {src_text}")
                print(f"   Ground Truth:  {tgt_text}")
                print(f"   Prediction:    {pred_text}")
                print(f"   Match: {'✅' if tgt_text.strip() == pred_text.strip() else '❌'}")
                print("-" * 50)
                
                examples_shown += 1
                if examples_shown >= num_examples:
                    break
# Load data
data = load_data()

# Create dataset
dataset = TransliterationDataset(
    data['src_tensor'], data['tgt_tensor'],
    data['src_lengths'], data['tgt_lengths']
)


# Create train/val/test splits (50%/25%/25%)
train_size = int(0.5 * len(dataset))
val_size = int(0.25 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)


print(f"📊 Dataset splits:")
print(f"   Train: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.1f}%)")
print(f"   Val:   {len(val_dataset)} ({len(val_dataset)/len(dataset)*100:.1f}%)")
print(f"   Test:  {len(test_dataset)} ({len(test_dataset)/len(dataset)*100:.1f}%)")



# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)



# Create model
model = Seq2SeqModel(
    src_vocab_size=len(data['vocab_src']),
    tgt_vocab_size=len(data['vocab_tgt']),
    embedding_dim=Config.EMBEDDING_DIM,
    hidden_size=Config.HIDDEN_SIZE,
    encoder_layers=Config.ENCODER_LAYERS,
    decoder_layers=Config.DECODER_LAYERS,
    dropout=Config.DROPOUT,
    src_pad_idx=data['PAD_SRC'],
    tgt_pad_idx=data['PAD_TGT']
).to(Config.DEVICE)



# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=data['PAD_TGT'])
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)



# Train model
train_losses, val_losses, test_metrics = train_model(
    model, train_loader, val_loader, test_loader,
    optimizer, criterion, Config.EPOCHS, Config.DEVICE,
    data['id2tgt'], freeze_encoder=True
)




test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
show_examples(model, test_loader, data['id2src'], data['id2tgt'], 
              Config.DEVICE, num_examples=10)



import os
import pickle
from google.colab import files
import zipfile
import json
from datetime import datetime

def save_model_checkpoint(model, optimizer, epoch, train_losses, val_losses, 
                         test_metrics, config, data_info, save_path="model_checkpoint.pt"):
    """
    Save complete model checkpoint with all necessary information
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': test_metrics,
        'config': config,
        'data_info': {
            'src_vocab_size': data_info['src_vocab_size'],
            'tgt_vocab_size': data_info['tgt_vocab_size'],
            'vocab_src': data_info['vocab_src'],
            'vocab_tgt': data_info['vocab_tgt'],
            'id2src': data_info['id2src'],
            'id2tgt': data_info['id2tgt'],
            'PAD_SRC': data_info['PAD_SRC'],
            'PAD_TGT': data_info['PAD_TGT']
        },
        'model_architecture': {
            'embedding_dim': Config.EMBEDDING_DIM,
            'hidden_size': Config.HIDDEN_SIZE,
            'encoder_layers': Config.ENCODER_LAYERS,
            'decoder_layers': Config.DECODER_LAYERS,
            'dropout': Config.DROPOUT
        },
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    torch.save(checkpoint, save_path)
    print(f"✅ Model checkpoint saved to: {save_path}")
    return save_path

def create_model_package(checkpoint_path, package_name="seq2seq_model_package"):
    """
    Create a complete package with model, config, and metadata
    """
    # Create package directory
    os.makedirs(package_name, exist_ok=True)
    
    # Load checkpoint to extract information
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Save model files
    torch.save(checkpoint, f"{package_name}/model_checkpoint.pt")
    
    # Save vocabularies separately for easy access
    with open(f"{package_name}/vocab_src.pkl", 'wb') as f:
        pickle.dump(checkpoint['data_info']['vocab_src'], f)
    
    with open(f"{package_name}/vocab_tgt.pkl", 'wb') as f:
        pickle.dump(checkpoint['data_info']['vocab_tgt'], f)
    
    # Save configuration as JSON
    config_info = {
        'model_architecture': checkpoint['model_architecture'],
        'training_config': checkpoint['config'],
        'data_info': {
            'src_vocab_size': checkpoint['data_info']['src_vocab_size'],
            'tgt_vocab_size': checkpoint['data_info']['tgt_vocab_size'],
            'PAD_SRC': checkpoint['data_info']['PAD_SRC'],
            'PAD_TGT': checkpoint['data_info']['PAD_TGT']
        },
        'performance': checkpoint['test_metrics'],
        'training_history': {
            'epochs_trained': checkpoint['epoch'],
            'final_train_loss': checkpoint['train_losses'][-1] if checkpoint['train_losses'] else None,
            'final_val_loss': checkpoint['val_losses'][-1] if checkpoint['val_losses'] else None
        },
        'timestamp': checkpoint['timestamp']
    }
    
    with open(f"{package_name}/model_info.json", 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    # Create README
    readme_content = f"""# Seq2Seq Urdu to Roman Transliteration Model

## Model Information
- **Architecture**: BiLSTM Encoder (2 layers) + LSTM Decoder (4 layers)
- **Embedding Dimension**: {checkpoint['model_architecture']['embedding_dim']}
- **Hidden Size**: {checkpoint['model_architecture']['hidden_size']}
- **Dropout**: {checkpoint['model_architecture']['dropout']}

## Performance Metrics
- **BLEU Score**: {checkpoint['test_metrics']['bleu']:.4f}
- **Perplexity**: {checkpoint['test_metrics']['perplexity']:.4f}
- **Character Error Rate**: {checkpoint['test_metrics']['cer']:.4f}

## Vocabulary Sizes
- **Source (Urdu)**: {checkpoint['data_info']['src_vocab_size']} tokens
- **Target (Roman)**: {checkpoint['data_info']['tgt_vocab_size']} tokens

## Training Details
- **Epochs Trained**: {checkpoint['epoch']}
- **Best Configuration**: LR={checkpoint['config']['lr']}, Batch Size={checkpoint['config']['batch_size']}
- **Timestamp**: {checkpoint['timestamp']}

## Files Included
- `model_checkpoint.pt`: Complete model checkpoint
- `vocab_src.pkl`: Source vocabulary (Urdu)
- `vocab_tgt.pkl`: Target vocabulary (Roman)
- `model_info.json`: Detailed configuration and metrics
- `README.md`: This file

## Usage
Load the model using the `load_model_checkpoint()` function.
"""
    
    with open(f"{package_name}/README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📦 Model package created: {package_name}/")
    return package_name

def download_model_package(package_name):
    """
    Create zip file and download model package
    """
    zip_filename = f"{package_name}.zip"
    
    # Create zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_name):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, package_name)
                zipf.write(file_path, arcname)
    
    print(f"📁 Created zip file: {zip_filename}")
    
    # Download file in Colab
    try:
        files.download(zip_filename)
        print(f"⬇️ Download initiated for: {zip_filename}")
    except:
        print(f"⚠️ Could not initiate download. File saved as: {zip_filename}")
    
    return zip_filename

def load_model_checkpoint(checkpoint_path, device=None):
    """
    Load model from checkpoint
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📂 Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model architecture
    model = Seq2SeqModel(
        src_vocab_size=checkpoint['data_info']['src_vocab_size'],
        tgt_vocab_size=checkpoint['data_info']['tgt_vocab_size'],
        embedding_dim=checkpoint['model_architecture']['embedding_dim'],
        hidden_size=checkpoint['model_architecture']['hidden_size'],
        encoder_layers=checkpoint['model_architecture']['encoder_layers'],
        decoder_layers=checkpoint['model_architecture']['decoder_layers'],
        dropout=checkpoint['model_architecture']['dropout'],
        src_pad_idx=checkpoint['data_info']['PAD_SRC'],
        tgt_pad_idx=checkpoint['data_info']['PAD_TGT']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer (if needed for further training)
    optimizer = optim.Adam(model.parameters(), lr=checkpoint['config']['lr'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print("✅ Model loaded successfully!")
    print(f"   - Trained for {checkpoint['epoch']} epochs")
    print(f"   - Test BLEU: {checkpoint['test_metrics']['bleu']:.4f}")
    print(f"   - Test Perplexity: {checkpoint['test_metrics']['perplexity']:.4f}")
    
    return {
        'model': model,
        'optimizer': optimizer,
        'checkpoint': checkpoint,
        'data_info': checkpoint['data_info']
    }




def continue_training_unfrozen(model, optimizer, train_loader, val_loader, test_loader,
                              criterion, additional_epochs, device, id2tgt, 
                              initial_epoch=0, prev_train_losses=None, prev_val_losses=None):
    """
    Continue training with unfrozen encoder
    """
    # Unfreeze all parameters
    print("🔓 Unfreezing all parameters - training both encoder and decoder")
    for param in model.parameters():
        param.requires_grad = True
    
    # Reduce learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1  # Reduce LR by 10x
        print(f"📉 Reduced learning rate to: {param_group['lr']}")
    
    train_losses = prev_train_losses if prev_train_losses else []
    val_losses = prev_val_losses if prev_val_losses else []
    
    print(f"\n🚀 Continuing training with unfrozen encoder")
    print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(additional_epochs):
        current_epoch = initial_epoch + epoch + 1
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_len']
            tgt_lengths = batch['tgt_len']
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(src, tgt, src_lengths, tgt_lengths)
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                           tgt[:, 1:].reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 50 == 0:  # More frequent logging for fine-tuning
                print(f"Epoch {current_epoch}/{initial_epoch + additional_epochs}, "
                      f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, id2tgt, device)
        val_losses.append(val_metrics['loss'])
        
        print(f"\n📈 Epoch {current_epoch}/{initial_epoch + additional_epochs} (Unfrozen)")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Val Perplexity: {val_metrics['perplexity']:.4f}")
        print(f"   Val BLEU: {val_metrics['bleu']:.4f}")
        print(f"   Val CER: {val_metrics['cer']:.4f}")
        print("-" * 50)
    
    # Final evaluation
    print("\n🧪 Final Test Evaluation (After Unfrozen Training):")
    test_metrics = evaluate_model(model, test_loader, criterion, id2tgt, device)
    print(f"   Test Loss: {test_metrics['loss']:.4f}")
    print(f"   Test Perplexity: {test_metrics['perplexity']:.4f}")
    print(f"   Test BLEU: {test_metrics['bleu']:.4f}")
    print(f"   Test CER: {test_metrics['cer']:.4f}")
    
    return train_losses, val_losses, test_metrics, current_epoch



def full_training_pipeline_with_save():
    """
    Complete training pipeline with model saving and continued training
    """
    # Load data (reuse from main function)
    data = load_data()
    
    # Create dataset and splits
    dataset = TransliterationDataset(
        data['src_tensor'], data['tgt_tensor'],
        data['src_lengths'], data['tgt_lengths']
    )
    
    train_size = int(0.5 * len(dataset))
    val_size = int(0.25 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Use best hyperparameters (you can modify these based on your results)
    best_lr = 1e-3
    best_batch_size = 32
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)
    
    # Create model
    model = Seq2SeqModel(
        src_vocab_size=len(data['vocab_src']),
        tgt_vocab_size=len(data['vocab_tgt']),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_size=Config.HIDDEN_SIZE,
        encoder_layers=Config.ENCODER_LAYERS,
        decoder_layers=Config.DECODER_LAYERS,
        dropout=Config.DROPOUT,
        src_pad_idx=data['PAD_SRC'],
        tgt_pad_idx=data['PAD_TGT']
    ).to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=data['PAD_TGT'])
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    
    print("=" * 60)
    print("🎯 PHASE 1: TRAINING WITH FROZEN ENCODER")
    print("=" * 60)
    
    # Phase 1: Train with frozen encoder
    train_losses, val_losses, test_metrics = train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, criterion, Config.EPOCHS, Config.DEVICE,
        data['id2tgt'], freeze_encoder=True
    )
    
    # Save checkpoint after Phase 1
    config_phase1 = {'lr': best_lr, 'batch_size': best_batch_size, 'phase': 'frozen_encoder'}
    data_info = {
        'src_vocab_size': len(data['vocab_src']),
        'tgt_vocab_size': len(data['vocab_tgt']),
        'vocab_src': data['vocab_src'],
        'vocab_tgt': data['vocab_tgt'],
        'id2src': data['id2src'],
        'id2tgt': data['id2tgt'],
        'PAD_SRC': data['PAD_SRC'],
        'PAD_TGT': data['PAD_TGT']
    }
    
    checkpoint_path = save_model_checkpoint(
        model, optimizer, Config.EPOCHS, train_losses, val_losses,
        test_metrics, config_phase1, data_info, "phase1_frozen_encoder.pt"
    )
    
    print("\n" + "=" * 60)
    print("🔥 PHASE 2: CONTINUE TRAINING WITH UNFROZEN ENCODER")
    print("=" * 60)
    
    # Phase 2: Continue training with unfrozen encoder
    additional_epochs = 20
    train_losses_unfrozen, val_losses_unfrozen, test_metrics_unfrozen, final_epoch = continue_training_unfrozen(
        model, optimizer, train_loader, val_loader, test_loader,
        criterion, additional_epochs, Config.DEVICE, data['id2tgt'],
        initial_epoch=Config.EPOCHS, prev_train_losses=train_losses, prev_val_losses=val_losses
    )
    
    # Save final checkpoint
    config_phase2 = {'lr': best_lr, 'batch_size': best_batch_size, 'phase': 'unfrozen_encoder'}
    final_checkpoint_path = save_model_checkpoint(
        model, optimizer, final_epoch, train_losses_unfrozen, val_losses_unfrozen,
        test_metrics_unfrozen, config_phase2, data_info, "final_unfrozen_model.pt"
    )
    
    # Create and download model package
    package_name = create_model_package(final_checkpoint_path, "seq2seq_urdu_roman_final")
    zip_file = download_model_package(package_name)
    
    # Show final examples
    print("\n" + "=" * 60)
    print("🎭 FINAL MODEL EXAMPLES")
    print("=" * 60)
    show_examples(model, test_loader, data['id2src'], data['id2tgt'], 
                  Config.DEVICE, num_examples=10)
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    print("Phase 1 (Frozen Encoder):")
    print(f"   BLEU: {test_metrics['bleu']:.4f}")
    print(f"   Perplexity: {test_metrics['perplexity']:.4f}")
    print(f"   CER: {test_metrics['cer']:.4f}")
    
    print("\nPhase 2 (Unfrozen Encoder):")
    print(f"   BLEU: {test_metrics_unfrozen['bleu']:.4f}")
    print(f"   Perplexity: {test_metrics_unfrozen['perplexity']:.4f}")
    print(f"   CER: {test_metrics_unfrozen['cer']:.4f}")
    
    improvement = test_metrics_unfrozen['bleu'] - test_metrics['bleu']
    print(f"\nBLEU Improvement: {improvement:+.4f}")
    
    return {
        'final_model': model,
        'phase1_metrics': test_metrics,
        'phase2_metrics': test_metrics_unfrozen,
        'package_path': package_name,
        'zip_file': zip_file
    }




# Run the complete pipeline
if __name__ == "__main__":
    # Run complete training pipeline with saving
    results = full_training_pipeline_with_save()
    
    print("\n🎉 Training complete!")
    print(f"📦 Model package saved and ready for download!")
    print(f"📁 Package location: {results['package_path']}")
    print(f"📄 Zip file: {results['zip_file']}")





# Example usage functions
def example_load_and_test():
    """
    Example of how to load a saved model and test it
    """
    # Load the saved model
    loaded_data = load_model_checkpoint("final_unfrozen_model.pt")
    model = loaded_data['model']
    data_info = loaded_data['data_info']
    
    print("🧪 Testing loaded model...")
    
    # You can now use the loaded model for inference
    # Example: translate a single sentence
    model.eval()
    
    return model, data_info






------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Neural Machine Translation: Urdu to Roman Transliteration
# Following NMT Architecture from PDF: 2-layer Encoder & 4-layer Decoder

# ====================================================================
# CELL 1: Setup and Installation
# ====================================================================

# Install required packages
!pip install nltk python-Levenshtein matplotlib torch torchvision torchaudio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein

# Download NLTK data
import nltk
nltk.download('punkt')

print("✅ All packages installed successfully!")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"💻 CUDA available: {torch.cuda.is_available()}")

# ====================================================================
# CELL 2: Configuration - Easy to Modify
# ====================================================================

class Config:
    """
    🎛️ Easy Configuration - Modify these values as needed!
    
    Architecture follows PDF specification:
    - Encoder: 2 LSTM layers (bidirectional)
    - Decoder: 4 LSTM layers (unidirectional)
    """
    # Data
    DATA_PATH = "/content/drive/MyDrive/TensorData/padded_dataset.pt"
    
    # Model Architecture (Easy to change!)
    EMBEDDING_DIM = 512      # Size of word embeddings
    HIDDEN_SIZE = 512        # LSTM hidden state size (can change to 20 as in PDF)
    ENCODER_LAYERS = 2       # PDF specifies 2 layers
    DECODER_LAYERS = 4       # PDF specifies 4 layers
    DROPOUT = 0.3           # Regularization
    
    # Training Parameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    
    # Phase 1: Frozen Encoder Training
    PHASE1_EPOCHS = 10      # Train decoder only
    
    # Phase 2: Full Training  
    PHASE2_EPOCHS = 20      # Train encoder + decoder
    PHASE2_LR_FACTOR = 0.1  # Reduce LR for fine-tuning
    
    # System
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
random.seed(Config.SEED)

print("⚙️ Configuration loaded!")
print(f"📱 Device: {Config.DEVICE}")
print(f"🏗️ Architecture: {Config.ENCODER_LAYERS}-layer Encoder → {Config.DECODER_LAYERS}-layer Decoder")
print(f"📏 Hidden Size: {Config.HIDDEN_SIZE}, Embedding: {Config.EMBEDDING_DIM}")

# ====================================================================
# CELL 3: Mount Google Drive and Load Data
# ====================================================================

from google.colab import drive
drive.mount('/content/drive')

def load_data():
    """
    📂 Load the preprocessed dataset
    
    Returns:
        dict: Contains all data components with their dimensions
    """
    print("📁 Loading dataset...")
    print(f"📍 Path: {Config.DATA_PATH}")
    
    try:
        data = torch.load(Config.DATA_PATH)
        print("✅ Data loaded successfully!")
    except FileNotFoundError:
        print("❌ Dataset not found! Please check the path.")
        return None
    
    # Extract components
    src_tensor = data["src_tensor"]     # Urdu tokens
    src_lengths = data["src_lengths"]
    tgt_tensor = data["tgt_tensor"]     # Roman tokens  
    tgt_lengths = data["tgt_lengths"]
    vocab_src = data["vocab_src"]       # Urdu vocabulary
    vocab_tgt = data["vocab_tgt"]       # Roman vocabulary
    
    # Create reverse mappings for decoding
    id2tgt = {i: token for token, i in vocab_tgt.items()}
    id2src = {i: token for token, i in vocab_src.items()}
    
    # Get special token indices
    PAD_SRC = vocab_src["<pad>"]
    PAD_TGT = vocab_tgt["<pad>"]
    SOS_TGT = vocab_tgt.get("<sos>", vocab_tgt.get("<start>", 1))
    EOS_TGT = vocab_tgt.get("<eos>", vocab_tgt.get("<end>", 2))
    
    print("\n📊 Dataset Information:")
    print(f"   📝 Number of samples: {src_tensor.size(0):,}")
    print(f"   📏 Max source length: {src_tensor.size(1)}")
    print(f"   📏 Max target length: {tgt_tensor.size(1)}")
    print(f"   🔤 Source vocab size: {len(vocab_src):,}")
    print(f"   🔤 Target vocab size: {len(vocab_tgt):,}")
    print(f"   🎯 Special tokens - PAD_SRC: {PAD_SRC}, PAD_TGT: {PAD_TGT}")
    print(f"   🎯 Special tokens - SOS_TGT: {SOS_TGT}, EOS_TGT: {EOS_TGT}")
    
    # Show tensor dimensions
    print(f"\n📐 Tensor Dimensions:")
    print(f"   📊 src_tensor: {src_tensor.shape}")
    print(f"   📊 tgt_tensor: {tgt_tensor.shape}")
    print(f"   📊 src_lengths: {src_lengths.shape}")
    print(f"   📊 tgt_lengths: {tgt_lengths.shape}")
    
    return {
        'src_tensor': src_tensor, 'src_lengths': src_lengths,
        'tgt_tensor': tgt_tensor, 'tgt_lengths': tgt_lengths,
        'vocab_src': vocab_src, 'vocab_tgt': vocab_tgt,
        'id2src': id2src, 'id2tgt': id2tgt,
        'PAD_SRC': PAD_SRC, 'PAD_TGT': PAD_TGT,
        'SOS_TGT': SOS_TGT, 'EOS_TGT': EOS_TGT
    }

# Load the data
data = load_data()

# ====================================================================
# CELL 4: Dataset Class
# ====================================================================

class TransliterationDataset(Dataset):
    """
    🗃️ Custom Dataset for Urdu-Roman transliteration
    
    Each sample contains:
    - src: Urdu sequence (input)
    - tgt: Roman sequence (target output) 
    - src_len: Actual length of source (excluding padding)
    - tgt_len: Actual length of target (excluding padding)
    """
    def __init__(self, src_tensor, tgt_tensor, src_lengths, tgt_lengths):
        self.src_tensor = src_tensor
        self.tgt_tensor = tgt_tensor
        self.src_lengths = src_lengths
        self.tgt_lengths = tgt_lengths
        
        print(f"📦 Dataset created with {len(self)} samples")
    
    def __len__(self):
        return len(self.src_tensor)
    
    def __getitem__(self, idx):
        return {
            'src': self.src_tensor[idx],      # Shape: (max_src_len,)
            'tgt': self.tgt_tensor[idx],      # Shape: (max_tgt_len,)  
            'src_len': self.src_lengths[idx], # Scalar
            'tgt_len': self.tgt_lengths[idx]  # Scalar
        }

# Create the dataset
dataset = TransliterationDataset(
    data['src_tensor'], data['tgt_tensor'],
    data['src_lengths'], data['tgt_lengths']
)

# Show a sample
print("\n🔍 Sample data point:")
sample = dataset[0]
print(f"   Source shape: {sample['src'].shape}")
print(f"   Target shape: {sample['tgt'].shape}")
print(f"   Source length: {sample['src_len']}")
print(f"   Target length: {sample['tgt_len']}")

# ====================================================================
# CELL 5: Data Splitting
# ====================================================================

def create_data_splits(dataset, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25):
    """
    🔄 Split dataset into train/validation/test sets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.SEED)
    )
    
    print("📊 Dataset splits:")
    print(f"   🚂 Train: {len(train_dataset):,} ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"   ✅ Validation: {len(val_dataset):,} ({len(val_dataset)/total_size*100:.1f}%)")
    print(f"   🧪 Test: {len(test_dataset):,} ({len(test_dataset)/total_size*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

# Create splits
train_dataset, val_dataset, test_dataset = create_data_splits(dataset)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

print(f"\n🔄 Data loaders created:")
print(f"   🚂 Train batches: {len(train_loader)}")
print(f"   ✅ Val batches: {len(val_loader)}")
print(f"   🧪 Test batches: {len(test_loader)}")

# Show batch dimensions
sample_batch = next(iter(train_loader))
print(f"\n📏 Batch dimensions:")
print(f"   📊 src batch: {sample_batch['src'].shape}")  # (batch_size, max_src_len)
print(f"   📊 tgt batch: {sample_batch['tgt'].shape}")  # (batch_size, max_tgt_len)

# ====================================================================
# CELL 6: Encoder Architecture (Following PDF Specifications)
# ====================================================================

class BiLSTMEncoder(nn.Module):
    """
    🔤 Bidirectional LSTM Encoder
    
    Architecture (from PDF):
    - 2 stacked LSTM layers 
    - Bidirectional (processes sequence left→right and right→left)
    - Projects bidirectional outputs to decoder hidden size
    
    Gradient Flow (PDF Rule):
    - Encoder parameters W(e) receive gradients through decoder initialization
    - Each layer receives gradients from layers above it
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, pad_idx):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        print(f"🏗️ Building Encoder:")
        print(f"   📝 Vocab size: {vocab_size:,}")
        print(f"   📏 Embedding dim: {embedding_dim}")
        print(f"   🧠 Hidden size: {hidden_size}")
        print(f"   🔢 Layers: {num_layers}")
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional LSTM (each direction has hidden_size units)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size,  # This is per direction
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,  # Creates 2*hidden_size outputs
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project bidirectional states (2*hidden_size) → decoder size (hidden_size)
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.cell_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        print(f"   ↔️ Bidirectional: True (output size: {hidden_size * 2})")
        print(f"   🎯 Projection: {hidden_size * 2} → {hidden_size}")
    
    def forward(self, src, src_lengths):
        """
        Forward pass with dimension tracking
        
        Input:
            src: (batch_size, max_src_len) - token indices
            src_lengths: (batch_size,) - actual sequence lengths
            
        Output:
            encoder_outputs: (batch_size, max_src_len, hidden_size*2) - all timestep outputs
            (hidden, cell): Each (num_layers, batch_size, hidden_size) - final states for decoder
        """
        batch_size = src.size(0)
        max_len = src.size(1)
        
        print(f"\n🔍 Encoder Forward Pass:")
        print(f"   📥 Input src: {src.shape}")
        print(f"   📥 Input lengths: {src_lengths.shape}")
        
        # Step 1: Embedding lookup
        embedded = self.dropout(self.embedding(src))
        print(f"   📝 After embedding: {embedded.shape}")
        
        # Step 2: Pack sequences for efficiency (handles variable lengths)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        print(f"   📦 Packed for efficiency (removes padding)")
        
        # Step 3: BiLSTM processing
        packed_output, (hidden, cell) = self.lstm(packed)
        print(f"   🧠 LSTM hidden states: {hidden.shape}")  # (num_layers*2, batch, hidden_size)
        print(f"   🧠 LSTM cell states: {cell.shape}")
        
        # Step 4: Unpack sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        print(f"   📤 Unpacked output: {output.shape}")  # (batch, max_len, hidden_size*2)
        
        # Step 5: Reshape bidirectional states
        # From: (num_layers*2, batch, hidden_size) → (num_layers, 2, batch, hidden_size)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        
        # Step 6: Concatenate forward and backward states
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)  # (num_layers, batch, hidden_size*2)
        cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)
        print(f"   ↔️ Concatenated states: {hidden.shape}")
        
        # Step 7: Project to decoder hidden size
        hidden = self.hidden_projection(hidden)  # (num_layers, batch, hidden_size)
        cell = self.cell_projection(cell)
        print(f"   🎯 Projected states: {hidden.shape}")
        
        return output, (hidden, cell)

# ====================================================================
# CELL 7: Decoder Architecture (Following PDF Specifications)  
# ====================================================================

class LSTMDecoder(nn.Module):
    """
    🎯 LSTM Decoder with Teacher Forcing
    
    Architecture (from PDF):
    - 4 stacked LSTM layers (unidirectional)
    - Takes previous target token as input
    - Outputs probability distribution over vocabulary
    
    Gradient Flow (PDF Rule):
    - Decoder parameters W(d) shape error signal ∂L/∂h
    - This error signal propagates back to encoder
    - Each layer receives gradients from output and layers above
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, pad_idx):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        print(f"🏗️ Building Decoder:")
        print(f"   📝 Vocab size: {vocab_size:,}")
        print(f"   📏 Embedding dim: {embedding_dim}")
        print(f"   🧠 Hidden size: {hidden_size}")
        print(f"   🔢 Layers: {num_layers}")
        
        # Word embeddings (shared vocabulary with encoder output)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        # 4-layer LSTM (as specified in PDF)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,  # Only embedding input (no attention)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection to vocabulary
        self.out = nn.Linear(hidden_size, vocab_size)
        print(f"   📤 Output projection: {hidden_size} → {vocab_size:,}")
    
    def forward(self, tgt, hidden, cell):
        """
        Forward pass with teacher forcing
        
        Input:
            tgt: (batch_size, seq_len) - target tokens (shifted input)
            hidden: (num_layers, batch_size, hidden_size) - initial hidden states
            cell: (num_layers, batch_size, hidden_size) - initial cell states
            
        Output:
            output: (batch_size, seq_len, vocab_size) - logits for each position
            (hidden, cell): Updated states for next timestep
        """
        print(f"\n🔍 Decoder Forward Pass:")
        print(f"   📥 Input tgt: {tgt.shape}")
        print(f"   📥 Hidden: {hidden.shape}")
        print(f"   📥 Cell: {cell.shape}")
        
        # Step 1: Embedding lookup
        embedded = self.dropout(self.embedding(tgt))
        print(f"   📝 After embedding: {embedded.shape}")
        
        # Step 2: LSTM processing
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        print(f"   🧠 LSTM output: {output.shape}")
        print(f"   🧠 Updated hidden: {hidden.shape}")
        
        # Step 3: Project to vocabulary
        output = self.out(output)
        print(f"   📤 Final logits: {output.shape}")
        
        return output, (hidden, cell)

# ====================================================================
# CELL 8: Complete Seq2Seq Model
# ====================================================================

class Seq2SeqModel(nn.Module):
    """
    🤖 Complete Sequence-to-Sequence Model
    
    Architecture:
    - BiLSTM Encoder (2 layers) → LSTM Decoder (4 layers)
    - Encoder final states initialize decoder states
    - Teacher forcing during training
    
    Following PDF gradient flow rules:
    - W(d) parameters shape error signal for encoder
    - Proper gradient flow through state initialization
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, hidden_size, 
                 encoder_layers, decoder_layers, dropout, src_pad_idx, tgt_pad_idx):
        super(Seq2SeqModel, self).__init__()
        
        print("🤖 Building Complete Seq2Seq Model:")
        print("="*50)
        
        # Build encoder and decoder
        self.encoder = BiLSTMEncoder(
            src_vocab_size, embedding_dim, hidden_size, 
            encoder_layers, dropout, src_pad_idx
        )
        
        print("\n" + "="*50)
        
        self.decoder = LSTMDecoder(
            tgt_vocab_size, embedding_dim, hidden_size,
            decoder_layers, dropout, tgt_pad_idx
        )
        
        # State initialization layers (encoder → decoder)
        self.init_hidden = nn.Linear(hidden_size, hidden_size)
        self.init_cell = nn.Linear(hidden_size, hidden_size)
        
        self.tgt_pad_idx = tgt_pad_idx
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        print(f"\n📊 Model Statistics:")
        print(f"   🔢 Total parameters: {total_params:,}")
        print(f"   📥 Encoder parameters: {encoder_params:,}")
        print(f"   📤 Decoder parameters: {decoder_params:,}")
        print(f"   🔄 Init parameters: {total_params - encoder_params - decoder_params:,}")
    
    def forward(self, src, tgt, src_lengths, tgt_lengths):
        """
        Complete forward pass
        
        Input:
            src: (batch_size, max_src_len) - source sequences
            tgt: (batch_size, max_tgt_len) - target sequences (with <sos>)
            src_lengths: (batch_size,) - actual source lengths
            tgt_lengths: (batch_size,) - actual target lengths
            
        Output:
            outputs: (batch_size, max_tgt_len-1, vocab_size) - predictions for each position
        """
        batch_size = src.size(0)
        
        print(f"\n🔄 Seq2Seq Forward Pass:")
        print(f"   📥 Source: {src.shape}")
        print(f"   📥 Target: {tgt.shape}")
        
        # Step 1: Encode source sequence
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(src, src_lengths)
        print(f"   🔤 Encoder outputs: {encoder_outputs.shape}")
        
        # Step 2: Initialize decoder states from encoder
        # Following PDF: decoder states initialized from encoder final states
        decoder_hidden = torch.zeros(self.decoder.num_layers, batch_size, 
                                   self.decoder.hidden_size, device=src.device)
        decoder_cell = torch.zeros(self.decoder.num_layers, batch_size, 
                                 self.decoder.hidden_size, device=src.device)
        
        # Initialize first layer with encoder states (other layers start at zero)
        decoder_hidden[0] = self.init_hidden(encoder_hidden[-1])  # Use final encoder layer
        decoder_cell[0] = self.init_cell(encoder_cell[-1])
        
        print(f"   🔄 Initialized decoder states: {decoder_hidden.shape}")
        
        # Step 3: Decode with teacher forcing
        # Use target sequence shifted by 1 (exclude last token for input)
        decoder_input = tgt[:, :-1]  # Remove last token
        target_output = tgt[:, 1:]   # Remove first token (<sos>)
        
        print(f"   📝 Decoder input: {decoder_input.shape}")
        print(f"   🎯 Target output: {target_output.shape}")
        
        # Step 4: Decoder forward pass
        outputs, _ = self.decoder(decoder_input, decoder_hidden, decoder_cell)
        print(f"   📤 Final outputs: {outputs.shape}")
        
        return outputs

# ====================================================================
# CELL 9: Initialize Model
# ====================================================================

# Create the model
print("🚀 Initializing Seq2Seq Model...")

model = Seq2SeqModel(
    src_vocab_size=len(data['vocab_src']),
    tgt_vocab_size=len(data['vocab_tgt']),
    embedding_dim=Config.EMBEDDING_DIM,
    hidden_size=Config.HIDDEN_SIZE,
    encoder_layers=Config.ENCODER_LAYERS,
    decoder_layers=Config.DECODER_LAYERS,
    dropout=Config.DROPOUT,
    src_pad_idx=data['PAD_SRC'],
    tgt_pad_idx=data['PAD_TGT']
).to(Config.DEVICE)

print(f"\n✅ Model created and moved to {Config.DEVICE}")

# Test model with a small batch
print("\n🧪 Testing model with sample batch...")
sample_batch = next(iter(train_loader))
src = sample_batch['src'].to(Config.DEVICE)
tgt = sample_batch['tgt'].to(Config.DEVICE)
src_lengths = sample_batch['src_len']
tgt_lengths = sample_batch['tgt_len']

with torch.no_grad():
    output = model(src, tgt, src_lengths, tgt_lengths)
    print(f"✅ Test successful! Output shape: {output.shape}")

# ====================================================================
# CELL 10: Evaluation Metrics
# ====================================================================

def calculate_accuracy(predictions, targets, id2tgt, pad_idx):
    """
    📊 Calculate exact match accuracy (sequence-level)
    """
    correct = 0
    total = 0
    
    for pred, target in zip(predictions, targets):
        # Convert to tokens and remove padding
        pred_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in pred]
        target_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in target]
        
        # Remove padding and special tokens
        pred_clean = [t for t in pred_tokens if t not in ['<pad>', '<sos>', '<eos>']]
        target_clean = [t for t in target_tokens if t not in ['<pad>', '<sos>', '<eos>']]
        
        # Exact match
        if pred_clean == target_clean:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0

def calculate_bleu(predictions, targets, id2tgt):
    """
    📊 Calculate BLEU score (n-gram overlap)
    """
    bleu_scores = []
    smooth_func = SmoothingFunction().method1
    
    for pred, target in zip(predictions, targets):
        pred_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in pred]
        target_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in target]
        
        # Clean tokens
        pred_tokens = [t for t in pred_tokens if t not in ['<pad>', '<sos>', '<eos>']]
        target_tokens = [t for t in target_tokens if t not in ['<pad>', '<sos>', '<eos>']]
        
        if len(pred_tokens) > 0 and len(target_tokens) > 0:
            score = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smooth_func)
            bleu_scores.append(score)
    
    return np.mean(bleu_scores) if bleu_scores else 0.0

def calculate_cer(predictions, targets, id2tgt):
    """
    📊 Calculate Character Error Rate using edit distance
    """
    distances = []
    
    for pred, target in zip(predictions, targets):
        pred_str = ''.join([id2tgt.get(idx.item(), '') for idx in pred])
        target_str = ''.join([id2tgt.get(idx.item(), '') for idx in target])
        
        # Clean strings
        pred_str = pred_str.replace('<pad>', '').replace('<sos>', '').replace('<eos>', '')
        target_str = target_str.replace('<pad>', '').replace('<sos>', '').replace('<eos>', '')
        
        if len(target_str) > 0:
            distance = Levenshtein.distance(pred_str, target_str) / len(target_str)
            distances.append(distance)
    
    return np.mean(distances) if distances else 1.0

def calculate_perplexity(loss):
    """
    📊 Calculate perplexity from cross-entropy loss
    """
    return math.exp(min(loss, 20))  # Cap to prevent overflow

print("✅ Evaluation metrics defined!")
print("   📊 Accuracy: Exact sequence match")
print("   📊 BLEU: N-gram overlap score")  
print("   📊 CER: Character-level edit distance")
print("   📊 Perplexity: Language model confidence")

# ====================================================================
# CELL 11: Model Evaluation Function
# ====================================================================

def evaluate_model(model, dataloader, criterion, id2tgt, device, desc="Evaluation"):
    """
    🧪 Comprehensive model evaluation
    
    Returns dictionary with all metrics
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    print(f"\n🧪 {desc}...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_len']
            tgt_lengths = batch['tgt_len']
            
            # Forward pass
            outputs = model(src, tgt, src_lengths, tgt_lengths)
            
            # Calculate loss (ignore padding)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                           tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            # Get predictions (argmax)
            predictions = torch.argmax(outputs, dim=-1)
            
            # Store for metric calculation
            all_predictions.extend(predictions.cpu())
            all_targets.extend(tgt[:, 1:].cpu())  # Skip <sos> token
            
            if batch_idx % 50 == 0:
                print(f"   📦 Processed {batch_idx}/{len(dataloader)} batches")
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    perplexity = calculate_perplexity(avg_loss)
    accuracy = calculate_accuracy(all_predictions, all_targets, id2tgt, data['PAD_TGT'])
    bleu = calculate_bleu(all_predictions, all_targets, id2tgt)
    cer = calculate_cer(all_predictions, all_targets, id2tgt)
    
    metrics = {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'bleu': bleu,
        'cer': cer
    }
    
    print(f"\n📊 {desc} Results:")
    print(f"   🔥 Loss: {avg_loss:.4f}")
    print(f"   🌟 Perplexity: {perplexity:.4f}")
    print(f"   🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   📝 BLEU: {bleu:.4f}")
    print(f"   📏 CER: {cer:.4f}")
    
    return metrics

print("✅ Evaluation function ready!")

# ====================================================================
# CELL 12: Training Function - Phase 1 (Frozen Encoder)
# ====================================================================

def train_phase1_frozen_encoder(model, train_loader, val_loader, optimizer, criterion, 
                                epochs, device, id2tgt):
    """
    🎯 Phase 1: Train decoder only (encoder frozen)
    
    Following PDF principle:
    - Only decoder parameters W(d) are updated
    - Encoder parameters W(e) are frozen
    - Decoder learns to use fixed encoder representations
    """
    print("\n" + "="*60)
    print("🔒 PHASE 1: TRAINING WITH FROZEN ENCODER")
    print("="*60)
    print("📚 Strategy: Train decoder to use pre-existing encoder features")
    print("🎯 Goal: Learn basic sequence-to-sequence mapping")
    
    # Freeze encoder parameters
    print("\n🔒 Freezing encoder parameters...")
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()
    
    print(f"   ❄️ Frozen parameters: {frozen_params:,}")
    print(f"   🔥 Trainable parameters: {trainable_params:,}")
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    print(f"\n🚀 Starting Phase 1 training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        print(f"\n📚 Epoch {epoch+1}/{epochs} - Training...")
        
        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_len']
            tgt_lengths = batch['tgt_len']
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(src, tgt, src_lengths, tgt_lengths)
            
            # Calculate loss (ignore padding tokens)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                           tgt[:, 1:].reshape(-1))
            
            # Backward pass (only decoder parameters updated)
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"   📊 Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation evaluation
        val_metrics = evaluate_model(model, val_loader, criterion, id2tgt, device, 
                                   f"Validation Epoch {epoch+1}")
        val_losses.append(val_metrics['loss'])
        val_metrics_history.append(val_metrics)
        
        # Print epoch summary
        print(f"\n📈 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   🚂 Train Loss: {avg_train_loss:.4f}")
        print(f"   ✅ Val Loss: {val_metrics['loss']:.4f}")
        print(f"   🎯 Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"   📝 Val BLEU: {val_metrics['bleu']:.4f}")
        print("-" * 50)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics_history': val_metrics_history,
        'final_val_metrics': val_metrics
    }

print("✅ Phase 1 training function ready!")

# ====================================================================
# CELL 13: Training Function - Phase 2 (Unfrozen Full Training)
# ====================================================================

def train_phase2_unfrozen(model, train_loader, val_loader, optimizer, criterion, 
                         epochs, device, id2tgt, lr_factor=0.1):
    """
    🔓 Phase 2: Train both encoder and decoder
    
    Following PDF principle:
    - All parameters (encoder + decoder) are updated
    - Decoder error signals flow back to encoder
    - Fine-tuning with reduced learning rate
    """
    print("\n" + "="*60)
    print("🔓 PHASE 2: FULL TRAINING (ENCODER + DECODER)")
    print("="*60)
    print("📚 Strategy: Fine-tune entire model end-to-end")
    print("🎯 Goal: Optimize encoder-decoder interaction")
    
    # Unfreeze all parameters
    print("\n🔓 Unfreezing all parameters...")
    total_params = 0
    
    for param in model.parameters():
        param.requires_grad = True
        total_params += param.numel()
    
    print(f"   🔥 Total trainable parameters: {total_params:,}")
    
    # Reduce learning rate for fine-tuning
    original_lr = optimizer.param_groups[0]['lr']
    new_lr = original_lr * lr_factor
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
        
    print(f"   📉 Learning rate: {original_lr:.2e} → {new_lr:.2e}")
    
    # Training tracking
    train_losses = []
    val_losses = []
    val_metrics_history = []
    
    print(f"\n🚀 Starting Phase 2 training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        print(f"\n📚 Epoch {epoch+1}/{epochs} - Full Training...")
        
        for batch_idx, batch in enumerate(train_loader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_len']
            tgt_lengths = batch['tgt_len']
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(src, tgt, src_lengths, tgt_lengths)
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                           tgt[:, 1:].reshape(-1))
            
            # Backward pass (all parameters updated)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 50 == 0:  # More frequent logging for fine-tuning
                print(f"   📊 Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation evaluation
        val_metrics = evaluate_model(model, val_loader, criterion, id2tgt, device,
                                   f"Validation Epoch {epoch+1} (Full)")
        val_losses.append(val_metrics['loss'])
        val_metrics_history.append(val_metrics)
        
        # Print epoch summary
        print(f"\n📈 Epoch {epoch+1}/{epochs} Summary (Full Training):")
        print(f"   🚂 Train Loss: {avg_train_loss:.4f}")
        print(f"   ✅ Val Loss: {val_metrics['loss']:.4f}")
        print(f"   🎯 Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"   📝 Val BLEU: {val_metrics['bleu']:.4f}")
        print("-" * 50)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics_history': val_metrics_history,
        'final_val_metrics': val_metrics
    }

print("✅ Phase 2 training function ready!")

# ====================================================================
# CELL 14: Initialize Training Components
# ====================================================================

# Loss function (ignores padding tokens)
criterion = nn.CrossEntropyLoss(ignore_index=data['PAD_TGT'])

# Optimizer (Adam with configured learning rate)
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

print("🛠️ Training components initialized:")
print(f"   📉 Loss function: CrossEntropyLoss (ignores pad token {data['PAD_TGT']})")
print(f"   🔧 Optimizer: Adam")
print(f"   📊 Learning rate: {Config.LEARNING_RATE}")
print(f"   🎯 Device: {Config.DEVICE}")

# Show model parameter breakdown
total_params = sum(p.numel() for p in model.parameters())
encoder_params = sum(p.numel() for p in model.encoder.parameters())
decoder_params = sum(p.numel() for p in model.decoder.parameters())

print(f"\n📊 Model Parameter Breakdown:")
print(f"   🔢 Total: {total_params:,}")
print(f"   📥 Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
print(f"   📤 Decoder: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
print(f"   🔄 Other: {total_params-encoder_params-decoder_params:,}")

# ====================================================================
# CELL 15: Run Phase 1 Training (Frozen Encoder)
# ====================================================================

print("🎬 Ready to start Phase 1 training!")
print(f"⚙️ Configuration:")
print(f"   📚 Epochs: {Config.PHASE1_EPOCHS}")
print(f"   📦 Batch size: {Config.BATCH_SIZE}")
print(f"   📊 Learning rate: {Config.LEARNING_RATE}")

# Run Phase 1 training
phase1_results = train_phase1_frozen_encoder(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=Config.PHASE1_EPOCHS,
    device=Config.DEVICE,
    id2tgt=data['id2tgt']
)

print("\n🎉 Phase 1 Training Complete!")
print(f"✅ Final validation loss: {phase1_results['final_val_metrics']['loss']:.4f}")
print(f"✅ Final validation accuracy: {phase1_results['final_val_metrics']['accuracy']:.4f}")

# ====================================================================
# CELL 16: Run Phase 2 Training (Full Training)
# ====================================================================

print("🎬 Ready to start Phase 2 training!")
print(f"⚙️ Configuration:")
print(f"   📚 Epochs: {Config.PHASE2_EPOCHS}")
print(f"   📉 LR reduction factor: {Config.PHASE2_LR_FACTOR}")

# Run Phase 2 training
phase2_results = train_phase2_unfrozen(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=Config.PHASE2_EPOCHS,
    device=Config.DEVICE,
    id2tgt=data['id2tgt'],
    lr_factor=Config.PHASE2_LR_FACTOR
)

print("\n🎉 Phase 2 Training Complete!")
print(f"✅ Final validation loss: {phase2_results['final_val_metrics']['loss']:.4f}")
print(f"✅ Final validation accuracy: {phase2_results['final_val_metrics']['accuracy']:.4f}")

# ====================================================================
# CELL 17: Final Test Evaluation
# ====================================================================

print("\n" + "="*60)
print("🧪 FINAL TEST EVALUATION")
print("="*60)

# Evaluate on test set
final_test_metrics = evaluate_model(
    model=model,
    dataloader=test_loader,
    criterion=criterion,
    id2tgt=data['id2tgt'],
    device=Config.DEVICE,
    desc="Final Test Evaluation"
)

# Compare phase results
print(f"\n📊 TRAINING PROGRESS SUMMARY")
print("="*60)
print(f"Phase 1 (Frozen Encoder):")
print(f"   🎯 Val Accuracy: {phase1_results['final_val_metrics']['accuracy']:.4f}")
print(f"   📝 Val BLEU: {phase1_results['final_val_metrics']['bleu']:.4f}")
print(f"   📏 Val CER: {phase1_results['final_val_metrics']['cer']:.4f}")

print(f"\nPhase 2 (Full Training):")
print(f"   🎯 Val Accuracy: {phase2_results['final_val_metrics']['accuracy']:.4f}")
print(f"   📝 Val BLEU: {phase2_results['final_val_metrics']['bleu']:.4f}")
print(f"   📏 Val CER: {phase2_results['final_val_metrics']['cer']:.4f}")

print(f"\nFinal Test Results:")
print(f"   🎯 Test Accuracy: {final_test_metrics['accuracy']:.4f}")
print(f"   📝 Test BLEU: {final_test_metrics['bleu']:.4f}")
print(f"   📏 Test CER: {final_test_metrics['cer']:.4f}")
print(f"   🌟 Test Perplexity: {final_test_metrics['perplexity']:.4f}")

# Calculate improvements
accuracy_improvement = phase2_results['final_val_metrics']['accuracy'] - phase1_results['final_val_metrics']['accuracy']
bleu_improvement = phase2_results['final_val_metrics']['bleu'] - phase1_results['final_val_metrics']['bleu']

print(f"\n📈 Phase 2 Improvements:")
print(f"   🎯 Accuracy: {accuracy_improvement:+.4f}")
print(f"   📝 BLEU: {bleu_improvement:+.4f}")

# ====================================================================
# CELL 18: Visualize Training Progress
# ====================================================================

def plot_training_curves(phase1_results, phase2_results):
    """
    📊 Plot training and validation curves for both phases
    """
    plt.figure(figsize=(15, 10))
    
    # Combine losses from both phases
    all_train_losses = phase1_results['train_losses'] + phase2_results['train_losses']
    all_val_losses = phase1_results['val_losses'] + phase2_results['val_losses']
    
    phase1_epochs = len(phase1_results['train_losses'])
    total_epochs = len(all_train_losses)
    epochs = list(range(1, total_epochs + 1))
    
    # Plot 1: Loss curves
    plt.subplot(2, 3, 1)
    plt.plot(epochs[:phase1_epochs], all_train_losses[:phase1_epochs], 'b-', label='Phase 1 Train', linewidth=2)
    plt.plot(epochs[:phase1_epochs], all_val_losses[:phase1_epochs], 'b--', label='Phase 1 Val', linewidth=2)
    plt.plot(epochs[phase1_epochs:], all_train_losses[phase1_epochs:], 'r-', label='Phase 2 Train', linewidth=2)
    plt.plot(epochs[phase1_epochs:], all_val_losses[phase1_epochs:], 'r--', label='Phase 2 Val', linewidth=2)
    plt.axvline(x=phase1_epochs, color='gray', linestyle=':', alpha=0.7, label='Phase Boundary')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy over time
    all_val_accuracies = [m['accuracy'] for m in phase1_results['val_metrics_history']] + \
                        [m['accuracy'] for m in phase2_results['val_metrics_history']]
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, all_val_accuracies, 'g-', linewidth=2, marker='o', markersize=4)
    plt.axvline(x=phase1_epochs, color='gray', linestyle=':', alpha=0.7, label='Phase Boundary')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: BLEU score over time
    all_val_bleu = [m['bleu'] for m in phase1_results['val_metrics_history']] + \
                   [m['bleu'] for m in phase2_results['val_metrics_history']]
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, all_val_bleu, 'purple', linewidth=2, marker='s', markersize=4)
    plt.axvline(x=phase1_epochs, color='gray', linestyle=':', alpha=0.7, label='Phase Boundary')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('Validation BLEU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: CER over time
    all_val_cer = [m['cer'] for m in phase1_results['val_metrics_history']] + \
                  [m['cer'] for m in phase2_results['val_metrics_history']]
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs, all_val_cer, 'orange', linewidth=2, marker='^', markersize=4)
    plt.axvline(x=phase1_epochs, color='gray', linestyle=':', alpha=0.7, label='Phase Boundary')
    plt.xlabel('Epoch')
    plt.ylabel('Character Error Rate')
    plt.title('Validation CER')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Perplexity over time
    all_val_perplexity = [m['perplexity'] for m in phase1_results['val_metrics_history']] + \
                         [m['perplexity'] for m in phase2_results['val_metrics_history']]
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs, all_val_perplexity, 'brown', linewidth=2, marker='d', markersize=4)
    plt.axvline(x=phase1_epochs, color='gray', linestyle=':', alpha=0.7, label='Phase Boundary')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Metrics comparison
    plt.subplot(2, 3, 6)
    metrics = ['Accuracy', 'BLEU', 'CER']
    phase1_vals = [phase1_results['final_val_metrics']['accuracy'],
                   phase1_results['final_val_metrics']['bleu'],
                   phase1_results['final_val_metrics']['cer']]
    phase2_vals = [phase2_results['final_val_metrics']['accuracy'],
                   phase2_results['final_val_metrics']['bleu'],
                   phase2_results['final_val_metrics']['cer']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, phase1_vals, width, label='Phase 1', alpha=0.8)
    plt.bar(x + width/2, phase2_vals, width, label='Phase 2', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Phase Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot the training curves
print("📊 Generating training visualization...")
plot_training_curves(phase1_results, phase2_results)

# ====================================================================
# CELL 19: Show Translation Examples
# ====================================================================

def show_translation_examples(model, test_loader, id2src, id2tgt, device, num_examples=10):
    """
    🔍 Show qualitative examples of translations
    """
    model.eval()
    examples_shown = 0
    
    print("\n🔍 Translation Examples:")
    print("=" * 80)
    
    with torch.no_grad():
        for batch in test_loader:
            if examples_shown >= num_examples:
                break
                
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lengths = batch['src_len']
            tgt_lengths = batch['tgt_len']
            
            # Get model predictions
            outputs = model(src, tgt, src_lengths, tgt_lengths)
            predictions = torch.argmax(outputs, dim=-1)
            
            batch_size = min(src.size(0), num_examples - examples_shown)
            
            for i in range(batch_size):
                # Convert indices to tokens
                src_indices = src[i][:src_lengths[i]]  # Remove padding
                tgt_indices = tgt[i, 1:tgt_lengths[i]]  # Remove <sos> and padding
                pred_indices = predictions[i][:tgt_lengths[i]-1]  # Match target length
                
                # Convert to readable text
                src_tokens = [id2src.get(idx.item(), '<unk>') for idx in src_indices]
                tgt_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in tgt_indices]
                pred_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in pred_indices]
                
                # Join tokens (remove special tokens for display)
                src_text = ' '.join([t for t in src_tokens if t not in ['<pad>', '<sos>', '<eos>']])
                tgt_text = ' '.join([t for t in tgt_tokens if t not in ['<pad>', '<sos>', '<eos>']])
                pred_text = ' '.join([t for t in pred_tokens if t not in ['<pad>', '<sos>', '<eos>']])
                
                # Check if prediction is correct
                is_correct = tgt_text.strip() == pred_text.strip()
                
                print(f"Example {examples_shown + 1}:")
                print(f"   📥 Source (Urdu): {src_text}")
                print(f"   🎯 Ground Truth:  {tgt_text}")
                print(f"   🤖 Prediction:    {pred_text}")
                print(f"   ✅ Match: {'✅ Correct' if is_correct else '❌ Incorrect'}")
                
                if not is_correct:
                    # Show character-level differences
                    cer = Levenshtein.distance(pred_text, tgt_text) / max(len(tgt_text), 1)
                    print(f"   📏 Character Error Rate: {cer:.3f}")
                
                print("-" * 60)
                
                examples_shown += 1
                if examples_shown >= num_examples:
                    break

# Show translation examples
show_translation_examples(
    model=model,
    test_loader=test_loader,
    id2src=data['id2src'],
    id2tgt=data['id2tgt'],
    device=Config.DEVICE,
    num_examples=10
)

# ====================================================================
# CELL 20: Model Analysis and Gradient Flow Verification
# ====================================================================

def analyze_gradient_flow(model, sample_batch, criterion):
    """
    🔍 Analyze gradient flow following PDF specifications
    
    Key insights from PDF:
    - W(d) parameters shape error signal ∂L/∂h
    - W(e) parameters receive gradients through decoder initialization
    - Gradient flow: Loss → Decoder → Encoder (through initialization)
    """
    model.train()  # Enable gradients
    
    print("🔍 Gradient Flow Analysis:")
    print("=" * 50)
    
    # Forward pass
    src = sample_batch['src'].to(Config.DEVICE)
    tgt = sample_batch['tgt'].to(Config.DEVICE)
    src_lengths = sample_batch['src_len']
    tgt_lengths = sample_batch['tgt_len']
    
    outputs = model(src, tgt, src_lengths, tgt_lengths)
    loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
    
    # Clear existing gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward()
    
    print(f"📊 Loss value: {loss.item():.4f}")
    print(f"\n🔄 Gradient Statistics:")
    
    # Analyze encoder gradients
    encoder_grad_norms = []
    decoder_grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            
            if 'encoder' in name:
                encoder_grad_norms.append(grad_norm)
                component = "📥 Encoder"
            elif 'decoder' in name:
                decoder_grad_norms.append(grad_norm)
                component = "📤 Decoder"
            else:
                component = "🔄 Other"
            
            print(f"   {component} {name}: {grad_norm:.6f}")
    
    # Summary statistics
    if encoder_grad_norms:
        avg_encoder_grad = np.mean(encoder_grad_norms)
        max_encoder_grad = np.max(encoder_grad_norms)
        print(f"\n📊 Encoder Gradient Summary:")
        print(f"   Average: {avg_encoder_grad:.6f}")
        print(f"   Maximum: {max_encoder_grad:.6f}")
    
    if decoder_grad_norms:
        avg_decoder_grad = np.mean(decoder_grad_norms)
        max_decoder_grad = np.max(decoder_grad_norms)
        print(f"\n📊 Decoder Gradient Summary:")
        print(f"   Average: {avg_decoder_grad:.6f}")
        print(f"   Maximum: {max_decoder_grad:.6f}")
    
    # Verify PDF principle
    print(f"\n✅ PDF Principle Verification:")
    print(f"   🔄 Decoder gradients flow to encoder: {'✅ Yes' if encoder_grad_norms else '❌ No'}")
    print(f"   📈 Error signal propagation: Loss → Decoder → Encoder")

# Analyze gradient flow with sample
sample_batch = next(iter(train_loader))
analyze_gradient_flow(model, sample_batch, criterion)

# ====================================================================
# CELL 21: Save Model and Results
# ====================================================================

import pickle
from datetime import datetime
import os

def save_training_results(model, optimizer, phase1_results, phase2_results, 
                         final_test_metrics, config, data_info):
    """
    💾 Save complete training results and model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results dictionary
    results = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'EMBEDDING_DIM': config.EMBEDDING_DIM,
            'HIDDEN_SIZE': config.HIDDEN_SIZE,
            'ENCODER_LAYERS': config.ENCODER_LAYERS,
            'DECODER_LAYERS': config.DECODER_LAYERS,
            'DROPOUT': config.DROPOUT,
            'LEARNING_RATE': config.LEARNING_RATE,
            'BATCH_SIZE': config.BATCH_SIZE,
            'PHASE1_EPOCHS': config.PHASE1_EPOCHS,
            'PHASE2_EPOCHS': config.PHASE2_EPOCHS,
            'PHASE2_LR_FACTOR': config.PHASE2_LR_FACTOR,
        },
        'data_info': data_info,
        'training_history': {
            'phase1': phase1_results,





