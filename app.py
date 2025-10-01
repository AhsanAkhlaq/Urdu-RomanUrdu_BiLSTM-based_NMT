import streamlit as st
import torch
import torch.nn as nn

# Set page config
st.set_page_config(
    page_title="Urdu to Roman Urdu Translator",
    page_icon="🔤",
    layout="wide"
)

# Constants
MAX_LENGTH = 50

# Model classes (same as your training code)
class Encoder(nn.Module):
    def __init__(self, vocab_size=512, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                             dropout=dropout, bidirectional=True, batch_first=True)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, (h, c) = self.bilstm(embedded)
        return output, (h, c)

class Decoder(nn.Module):
    def __init__(self, vocab_size=512, embed_dim=256, hidden_dim=256, num_layers=4,
                 output_vocab_size=512, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                           dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_vocab_size)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Project encoder states to decoder dimensions
        self.h_projection = nn.Linear(256, hidden_dim)  # 256 from bidirectional encoder
        self.c_projection = nn.Linear(256, hidden_dim)

    def forward(self, x, encoder_states):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        h_enc, c_enc = encoder_states
        # Convert bidirectional encoder states to decoder format
        # h_enc: [num_layers*2, batch, hidden_dim] -> [num_layers, batch, hidden_dim*2]
        batch_size = h_enc.size(1)
        h_enc = h_enc.view(2, 2, batch_size, -1)  # [directions, layers, batch, hidden]
        c_enc = c_enc.view(2, 2, batch_size, -1)

        # Concatenate forward and backward states
        h_enc = torch.cat([h_enc[0], h_enc[1]], dim=-1)  # [layers, batch, hidden*2]
        c_enc = torch.cat([c_enc[0], c_enc[1]], dim=-1)

        # Project to decoder dimensions and repeat for all decoder layers
        h_init = self.h_projection(h_enc[-1]).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_init = self.c_projection(c_enc[-1]).unsqueeze(0).repeat(self.num_layers, 1, 1)

        initial_state = (h_init, c_init)

        output, _ = self.lstm(embedded, initial_state)
        output = self.linear(output)
        return output

class Seq2SeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src):
        # Encoder processes source
        enc_output, enc_states = self.encoder(src)
        # Decoder processes same source sequence
        dec_output = self.decoder(src, enc_states)
        return dec_output

# Tokenization functions
def encode_sentence(sentence, token2id, is_urdu=True, max_length=MAX_LENGTH):
    """Encode a sentence using greedy longest-match subword tokenization and pad to max_length."""
    tokens = []

    if is_urdu:
        words = ["_" + w for w in sentence.split()]
    else:
        words = [w + "_" for w in sentence.split()]

    for w in words:
        i = 0
        while i < len(w):
            subword = None
            for j in range(len(w), i, -1):
                piece = w[i:j]
                if piece in token2id:
                    subword = piece
                    break
            if subword is None:
                tokens.append(token2id.get("<unk>", 3))  # Default unk token
                i += 1
            else:
                tokens.append(token2id[subword])
                i += len(subword)

    # Add SOS and EOS tokens
    encoded = [token2id.get("<sos>", 1)] + tokens + [token2id.get("<eos>", 2)]
    
    # Truncate if longer than max_length
    if len(encoded) > max_length:
        encoded = encoded[:max_length-1] + [token2id.get("<eos>", 2)]
    
    # Pad to exactly max_length with PAD token (0)
    while len(encoded) < max_length:
        encoded.append(0)  # PAD token
    
    return encoded

def decode_tokens(tokens, id2token):
    """Convert token IDs back to text"""
    words = []
    for token_id in tokens:
        if token_id in [0, 1, 2]:  # pad, sos, eos
            continue
        words.append(id2token.get(token_id, '<unk>'))
    text = ''.join(words)
    # Replace BPE markers with spaces (_ represents actual spaces)
    text = text.replace('_', ' ').strip()
    # Remove extra spaces
    text = ' '.join(text.split())
    return text


# Cached functions for loading resources
@st.cache_resource
def load_model_and_vocabularies(model_path):
    """Load trained model and vocabularies from the saved checkpoint"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract vocabularies from checkpoint
        vocab_urdu = checkpoint['urdu_vocab']
        vocab_roman = checkpoint['roman_vocab']
        
        # Build reverse maps
        id2urdu = {i: t for t, i in vocab_urdu.items()}
        id2roman = {i: t for t, i in vocab_roman.items()}
        
        # Create and load model
        model = Seq2SeqModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, device, vocab_urdu, vocab_roman, id2urdu, id2roman
        
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}")
        st.error("Please make sure urdu_roman_nmt_model.pth is in the Model2/ directory")
        return None, None, None, None, None, None
    except KeyError as e:
        st.error(f"Missing key in checkpoint: {e}")
        st.error("The model file doesn't contain the expected keys (model_state_dict, urdu_vocab, roman_vocab)")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None

def translate_text(text, model, vocab_urdu, vocab_roman, id2roman, device):
    """Translate Urdu text to Roman Urdu"""
    try:
        # Encode the input text with padding to MAX_LENGTH
        encoded = encode_sentence(text, vocab_urdu, is_urdu=True, max_length=MAX_LENGTH)
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=-1).squeeze(0)
        
        # Decode the prediction
        translated_text = decode_tokens(prediction.cpu().numpy(), id2roman)
        
        return translated_text, encoded, prediction.cpu().numpy()
    
    except Exception as e:
        st.error(f"Translation error: {e}")
        return None, None, None

# Main Streamlit App
def main():
    st.title("🔤 Urdu to Roman Urdu Translator")
    st.markdown("### Neural Machine Translation using Seq2Seq with LSTM")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("Configuration")
        
        model_path = "Model2/urdu_roman_nmt_model.pth"
        st.text_input(
            "Model Path:", 
            value=model_path,
            disabled=True,
            help="Static path to your trained model file"
        )
        
        st.markdown("---")
        st.markdown("**Model Info:**")
        st.markdown("- Encoder: BiLSTM (2 layers)")
        st.markdown("- Decoder: LSTM (4 layers)")
        st.markdown("- Tokenization: BPE Subwords")
        st.markdown(f"- Max Length: {MAX_LENGTH} tokens")
        st.markdown("- Input: Padded to exact length")
        
        st.markdown("---")
        st.markdown("**Metrics:**")
        show_metrics = st.checkbox("Show token analysis", value=False)
    
    # Load model and vocabularies
    result = load_model_and_vocabularies(model_path)
    
    if result[0] is None:  # Check if loading failed
        st.stop()
    
    model, device, vocab_urdu, vocab_roman, id2urdu, id2roman = result
    
    st.success(f"✅ Model and vocabularies loaded successfully on {device}")
    
    # Display vocabulary info
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.info(f"📚 Urdu vocabulary size: {len(vocab_urdu)}")
    with col_info2:
        st.info(f"🔤 Roman vocabulary size: {len(vocab_roman)}")
    with col_info3:
        st.info(f"📏 Max sequence length: {MAX_LENGTH}")
    
    # Main translation interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input (Urdu)")
        
        # Input methods
        input_method = st.radio("Input method:", ["Type text", "Example sentences"])
        
        if input_method == "Type text":
            urdu_text = st.text_area(
                "Enter Urdu text:",
                height=150,
                placeholder="یہاں اردو متن لکھیں..."
            )
        else:
            example_sentences = [
                "میں تم سے محبت کرتا ہوں",
                "تو کہ یکتا تھا بے شمار ہوا",
                "ہم بھی ٹوٹیں تو جا بجا ہو جائیں",
                "ہم بھی مجبوریوں کا عذر کریں",
                "پھر کہیں اور مبتلا ہو جائیں",
                "ہم اگر منزلیں نہ بن پائے",
                "منزلوں تک کا راستا ہو جائیں",
                "دیر سے سوچ میں ہیں پروانے",
                "راکھ ہو جائیں یا ہوا ہو جائیں",
                "عشق بھی کھیل ہے نصیبوں کا",
                "خاک ہو جائیں کیمیا ہو جائیں",
                "اب کے گر تو ملے تو ہم تجھ سے",
                "ایسے لپٹیں تری قبا ہو جائیں",
                "بندگی ہم نے چھوڑ دی ہے فرازؔ",
                "کیا کریں لوگ جب خدا ہو جائیں",
                "جب بھی دل کھول کے روئے ہوں گے",
                "لوگ آرام سے سوئے ہوں گے",
                "بعض اوقات بہ مجبوریٔ دل",
                "ہم تو کیا آپ بھی روئے ہوں گے",
                "صبح تک دست صبا نے کیا کیا"
            ]
            
            selected_example = st.selectbox("Choose an example:", example_sentences)
            urdu_text = selected_example
            
            st.text_area("Selected text:", value=urdu_text, height=100, disabled=True)
    
    with col2:
        st.subheader("🔤 Output (Roman Urdu)")
        
        if st.button("🚀 Translate", type="primary"):
            if urdu_text.strip():
                with st.spinner("Translating..."):
                    translated_text, encoded_tokens, predicted_tokens = translate_text(
                        urdu_text, model, vocab_urdu, vocab_roman, id2roman, device
                    )
                
                if translated_text:
                    st.text_area(
                        "Translation:",
                        value=translated_text,
                        height=150,
                        disabled=True
                    )
                    
                    # Show token analysis
                    if show_metrics and translated_text:
                        with st.expander("🔍 Token Analysis"):
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.markdown("**Input Tokens:**")
                                st.write(f"Total Count: {len(encoded_tokens)} (padded to {MAX_LENGTH})")
                                # Count non-padding tokens
                                non_pad = sum(1 for t in encoded_tokens if t != 0)
                                st.write(f"Non-padding: {non_pad}")
                                # Show tokens in a more readable format
                                token_display = [f"{i}: {id2urdu.get(token, '<unk>' if token != 0 else '<PAD>')}" 
                                               for i, token in enumerate(encoded_tokens[:20])]
                                for token_info in token_display:
                                    st.caption(token_info)
                                if len(encoded_tokens) > 20:
                                    st.caption(f"... and {len(encoded_tokens) - 20} more tokens")
                            
                            with col_b:
                                st.markdown("**Output Tokens:**")
                                st.write(f"Total Count: {len(predicted_tokens)}")
                                # Count non-padding tokens
                                non_pad = sum(1 for t in predicted_tokens if t != 0)
                                st.write(f"Non-padding: {non_pad}")
                                # Show tokens in a more readable format
                                token_display = [f"{i}: {id2roman.get(token, '<unk>' if token != 0 else '<PAD>')}" 
                                               for i, token in enumerate(predicted_tokens[:20])]
                                for token_info in token_display:
                                    st.caption(token_info)
                                if len(predicted_tokens) > 20:
                                    st.caption(f"... and {len(predicted_tokens) - 20} more tokens")
                
                else:
                    st.error("Translation failed. Please try again.")
            else:
                st.warning("Please enter some Urdu text to translate.")
    
    # Information section
    with st.expander("ℹ️ About This App"):
        st.markdown(f"""
        This Neural Machine Translation system converts Urdu text to Roman Urdu using:
        
        **Architecture:**
        - **Encoder**: Bidirectional LSTM with 2 layers
        - **Decoder**: Unidirectional LSTM with 4 layers
        - **Tokenization**: BPE (Byte Pair Encoding) subwords
        - **Sequence Length**: Fixed at {MAX_LENGTH} tokens (padded)
        
        **Features:**
        - Real-time translation
        - Fixed-length input sequences (padded to {MAX_LENGTH})
        - Token-level analysis
        - Automatic truncation for long sentences
        
        **Usage Tips:**
        - For best results, use complete sentences
        - Long sentences are automatically truncated to {MAX_LENGTH} tokens
        - Short sentences are padded to exactly {MAX_LENGTH} tokens
        - The model works with BPE tokenization
        """)

if __name__ == "__main__":
    main()