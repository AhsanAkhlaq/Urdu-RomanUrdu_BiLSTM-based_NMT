import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import random

#C:\Users\NCS\anaconda3\envs\env_ML_2\python.exe -m streamlit run app.py
# Set page config
st.set_page_config(
    page_title="Urdu to Roman Urdu Translator",
    page_icon="🔤",
    layout="wide"
)

# ====================================================================
# Model Architecture Classes (Required for loading saved models)
# ====================================================================

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, pad_idx):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, src, src_lengths):
        batch_size = src.size(0)
        embedded = self.dropout(self.embedding(src))
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Reshape bidirectional states
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        
        # Concatenate forward and backward states
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)
        cell = torch.cat((cell[:, 0], cell[:, 1]), dim=2)
        
        return output, (hidden, cell)

class LSTMDecoder(nn.Module):
    """
    LSTM Decoder with Teacher Forcing
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, pad_idx):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection to vocabulary
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, hidden, cell):
        embedded = self.dropout(self.embedding(tgt))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.out(output)
        return output, (hidden, cell)

class Seq2SeqModel(nn.Module):
    """
    Complete Sequence-to-Sequence Model
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, hidden_size,
                 encoder_layers, decoder_layers, dropout, src_pad_idx, tgt_pad_idx):
        super(Seq2SeqModel, self).__init__()

        # Build encoder and decoder
        self.encoder = BiLSTMEncoder(
            src_vocab_size, embedding_dim, hidden_size,
            encoder_layers, dropout, src_pad_idx
        )

        self.decoder = LSTMDecoder(
            tgt_vocab_size, embedding_dim, hidden_size*2,
            decoder_layers, dropout, tgt_pad_idx
        )

        # State initialization layers (encoder → decoder)
        self.init_hidden = nn.Linear(hidden_size*2, hidden_size*2)
        self.init_cell = nn.Linear(hidden_size*2, hidden_size*2)

        self.tgt_pad_idx = tgt_pad_idx

    def forward(self, src, src_lengths, teacher_forcing_ratio=1.0):
        batch_size = src.size(0)
        max_tgt_len = 50
        vocab_size = self.decoder.vocab_size
        device = src.device

        # Encode source sequence
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder(src, src_lengths)

        # Initialize decoder states from encoder
        decoder_hidden = torch.zeros(self.decoder.num_layers, batch_size,
                                   self.decoder.hidden_size, device=device)
        decoder_cell = torch.zeros(self.decoder.num_layers, batch_size,
                                 self.decoder.hidden_size, device=device)

        decoder_hidden[0] = self.init_hidden(encoder_hidden[-1])
        decoder_cell[0] = self.init_cell(encoder_cell[-1])

        # Decode with teacher forcing ratio
        outputs = torch.zeros(batch_size, max_tgt_len, vocab_size, device=device)
        
        padded_src = torch.full((batch_size, max_tgt_len+1), self.tgt_pad_idx, device=device)
        padded_src[:, :src.size(1)] = src
        
        decoder_input = padded_src[:, 0].unsqueeze(1)

        for t in range(max_tgt_len):
            output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input, decoder_hidden, decoder_cell
            )
            outputs[:, t:t+1, :] = output

            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing and t < max_tgt_len - 1:
                decoder_input = padded_src[:, t+1].unsqueeze(1)
            else:
                predicted_token = output.argmax(dim=-1)
                decoder_input = predicted_token

        return outputs

# Encoding functions (from your training code)
def encode_sentence(sentence, token2id, is_urdu=True):
    """
    Encode a sentence using greedy longest-match subword tokenization.
    - `token2id` is your BPE subword vocab (already contains things like "mohabbat_").
    - For Urdu: prepend "_" to each word.
    - For Roman: append "_" to each word.
    """
    tokens = []
    if is_urdu:
        words = ["_" + w for w in sentence.split()]   # Urdu side
    else:
        words = [w + "_" for w in sentence.split()]   # Roman side
    
    for w in words:
        i = 0
        while i < len(w):
            # try to find the longest subword starting at position i
            subword = None
            for j in range(len(w), i, -1):
                piece = w[i:j]
                if piece in token2id:
                    subword = piece
                    break
            if subword is None:
                tokens.append(token2id.get("<unk>", 0))
                i += 1
            else:
                tokens.append(token2id[subword])
                i += len(subword)
    
    # add special tokens
    return [token2id.get("<sos>", 1)] + tokens + [token2id.get("<eos>", 2)]

def decode_sentence(token_ids, id2token):
    """
    Decode a sequence of token IDs back to text.
    Handles BPE subword tokens properly.
    """
    tokens = []
    for token_id in token_ids:
        if token_id in id2token:
            token = id2token[token_id]
            if token not in ["<sos>", "<eos>", "<pad>", "<unk>"]:
                tokens.append(token)
    
    # Join tokens to reconstruct words
    text = "".join(tokens)
    
    # For Roman Urdu: underscores are at the end of words (word_)
    # For Urdu: underscores are at the beginning of words (_word)
    
    # Handle Roman Urdu format (word_)
    if text and '_' in text:
        # Split by underscore and clean up
        words = []
        current_word = ""
        
        for char in text:
            if char == '_':
                if current_word:
                    words.append(current_word)
                    current_word = ""
            else:
                current_word += char
        
        # Add the last word if it doesn't end with underscore
        if current_word:
            words.append(current_word)
        
        text = " ".join(words)
    
    # Clean up extra spaces
    text = " ".join(text.split())
    return text.strip()

def create_padding_mask(sequences, pad_idx):
    """
    Create padding mask for sequences
    Returns: (batch_size, seq_len) - True for padded positions
    """
    return sequences == pad_idx

def get_sequence_lengths(sequences, pad_idx):
    """
    Get actual sequence lengths (excluding padding)
    """
    mask = sequences != pad_idx
    lengths = mask.sum(dim=1)
    return lengths

@st.cache_resource
def load_model(model_path):
    """Load the trained model and vocabularies"""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please check the file path.")
            return None, None, None, None, None
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract vocabularies and config
        vocab_src = checkpoint['vocab_src']  # Urdu vocab
        vocab_tgt = checkpoint['vocab_tgt']  # Roman vocab
        id2src = checkpoint['id2src']
        id2tgt = checkpoint['id2tgt']
        config = checkpoint['config']
        
        # Create model from config instead of loading full object
        model = Seq2SeqModel(
            src_vocab_size=len(vocab_src),
            tgt_vocab_size=len(vocab_tgt),
            embedding_dim=config['embedding_dim'],
            hidden_size=config['hidden_size'],
            encoder_layers=config['encoder_layers'],
            decoder_layers=config['decoder_layers'],
            dropout=config['dropout'],
            src_pad_idx=vocab_src.get('<pad>', 0),
            tgt_pad_idx=vocab_tgt.get('<pad>', 0)
        )
        
        # Load the state dict (weights only)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        
        return model, vocab_src, vocab_tgt, id2src, id2tgt
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

def get_model_files():
    """Get list of .pt files in current directory and subdirectories"""
    model_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pt'):
                model_files.append(os.path.join(root, file))
    return model_files

def translate_text(model, text, vocab_src, vocab_tgt, id2src, id2tgt, max_length=50):
    """
    Translate Urdu text to Roman Urdu using proper inference procedure
    """
    try:
        # Step 1: Encode input sentence (Urdu)
        input_ids = encode_sentence(text.strip(), vocab_src, is_urdu=True)
        
        # Convert to tensor and add batch dimension
        src_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, src_len)
        src_lengths = torch.tensor([len(input_ids)], dtype=torch.long)
        
        # Debug info
        st.write(f"🔍 **Debug Info:**")
        st.write(f"- Input text: '{text.strip()}'")
        st.write(f"- Encoded tokens: {input_ids}    ")
        st.write(f"- Sequence length: {len(input_ids)}")
        
        with torch.no_grad():
            model.eval()  # Ensure model is in eval mode
            # Get model predictions
            outputs = model(src_tensor, src_lengths, teacher_forcing_ratio=1)
            predictions = torch.argmax(outputs, dim=-1)


            pred_indices = predictions[:50].squeeze(0)  # (max_tgt_len,)

            # Convert to readable text
            pred_tokens = [id2tgt.get(idx.item(), '<unk>') for idx in pred_indices]

            # Join tokens (remove special tokens for display)
            pred_text = ' '.join([t for t in pred_tokens if t not in ['<pad>', '<sos>', '<eos>']])


            st.write(f"- Final translation: '{pred_text}'")
            
            return pred_text if pred_text.strip() else "Empty translation"
    
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return "Translation failed"

def main():
    st.title("🔤 Urdu to Roman Urdu Translator")
    st.markdown("---")
    
    # Initialize session state for model
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.vocab_src = None
        st.session_state.vocab_tgt = None
        st.session_state.id2src = None
        st.session_state.id2tgt = None
    
    # Model Selection Section
    st.header("📁 Model Selection")
    
    # Create tabs for different selection methods
    tab1, tab2, tab3 = st.tabs(["📂 Browse Files", "📝 Enter Path", "🔍 Auto-detect"])
    
    with tab1:
        st.subheader("Upload Model File")
        uploaded_file = st.file_uploader(
            "Choose your model file (.pt)",
            type=['pt'],
            help="Upload your trained PyTorch model file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if st.button("Load Uploaded Model", key="load_uploaded"):
                with st.spinner("Loading model..."):
                    model, vocab_src, vocab_tgt, id2src, id2tgt = load_model(temp_path)
                
                if model is not None:
                    st.session_state.model_loaded = True
                    st.session_state.model = model
                    st.session_state.vocab_src = vocab_src
                    st.session_state.vocab_tgt = vocab_tgt
                    st.session_state.id2src = id2src
                    st.session_state.id2tgt = id2tgt
                    st.success(f"✅ Model loaded successfully from: {uploaded_file.name}")
                    st.rerun()
    
    with tab2:
        st.subheader("Enter Model Path")
        model_path = st.text_input(
            "Model file path:",
            placeholder="e.g., ./models/nmt_app_model.pt",
            help="Enter the full path to your model file"
        )
        
        if model_path and st.button("Load Model from Path", key="load_path"):
            with st.spinner("Loading model..."):
                model, vocab_src, vocab_tgt, id2src, id2tgt = load_model(model_path)
            
            if model is not None:
                st.session_state.model_loaded = True
                st.session_state.model = model
                st.session_state.vocab_src = vocab_src
                st.session_state.vocab_tgt = vocab_tgt
                st.session_state.id2src = id2src
                st.session_state.id2tgt = id2tgt
                st.success(f"✅ Model loaded successfully from: {model_path}")
                st.rerun()
    
    with tab3:
        st.subheader("Auto-detect Models")
        model_files = get_model_files()
        
        if model_files:
            selected_model = st.selectbox(
                "Select a model file:",
                model_files,
                help="Choose from detected .pt files in current directory and subdirectories"
            )
            
            if st.button("Load Selected Model", key="load_selected"):
                with st.spinner("Loading model..."):
                    model, vocab_src, vocab_tgt, id2src, id2tgt = load_model(selected_model)
                
                if model is not None:
                    st.session_state.model_loaded = True
                    st.session_state.model = model
                    st.session_state.vocab_src = vocab_src
                    st.session_state.vocab_tgt = vocab_tgt
                    st.session_state.id2src = id2src
                    st.session_state.id2tgt = id2tgt
                    st.success(f"✅ Model loaded successfully from: {selected_model}")
                    st.rerun()
        else:
            st.info("No .pt files found in the current directory and subdirectories.")
    
    # Show current model status
    if st.session_state.model_loaded:
        st.success("🚀 Model is ready for translation!")
        if st.button("🔄 Change Model", key="change_model"):
            st.session_state.model_loaded = False
            st.rerun()
    else:
        st.warning("⚠️ Please load a model to start translation.")
        st.stop()
    
    st.markdown("---")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📝 Urdu Input")
        urdu_text = st.text_area(
            "Enter Urdu text:",
            height=200,
            placeholder="یہاں اردو متن لکھیں...",
            help="Type or paste your Urdu text here"
        )
        
        # Translation button
        translate_btn = st.button("🔄 Translate", type="primary", use_container_width=True)
    
    with col2:
        st.header("🔤 Roman Urdu Output")
        
        if translate_btn and urdu_text.strip():
            with st.spinner("Translating..."):
                translation = translate_text(
                    st.session_state.model, urdu_text.strip(), 
                    st.session_state.vocab_src, st.session_state.vocab_tgt, 
                    st.session_state.id2src, st.session_state.id2tgt
                )
            
            st.text_area(
                "Translation:",
                value=translation,
                height=200,
                disabled=True
            )
            
            # Copy button (using st.code for easy copying)
            st.code(translation, language=None)
        
        elif translate_btn and not urdu_text.strip():
            st.warning("Please enter some Urdu text to translate.")
        else:
            st.text_area(
                "Translation will appear here:",
                value="",
                height=200,
                disabled=True,
                placeholder="Translation will appear here after clicking translate..."
            )
    
    # Examples section
    st.markdown("---")
    st.header("📚 Example Translations")
    
    examples = [
        "یہ ایک اچھا دن ہے",
        "میرا نام علی ہے",
        "آپ کیسے ہیں؟",
        "شکریہ آپ کا"
    ]
    
    example_cols = st.columns(len(examples))
    
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(f"Try: {example}", key=f"example_{i}"):
                st.session_state.example_text = example
    
    # If example was clicked, display it in the text area
    if 'example_text' in st.session_state:
        urdu_text = st.session_state.example_text
        del st.session_state.example_text  # Clear after use
    
    # Model info
    st.markdown("---")
    st.header("ℹ️ Model Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info(f"""
        **Model Details:**
        - Source Language: Urdu
        - Target Language: Roman Urdu
        - Model Type: Sequence-to-Sequence
        """)
    
    with info_col2:
        if st.session_state.model_loaded:
            st.info(f"""
            **Vocabulary Size:**
            - Urdu Tokens: {len(st.session_state.vocab_src)}
            - Roman Tokens: {len(st.session_state.vocab_tgt)}
            - Model Status: Ready
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit • Urdu ↔ Roman Translation"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()