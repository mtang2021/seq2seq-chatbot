import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import pickle

# Define constants
pad_word = "<pad>"
bos_word = "<s>"
eos_word = "</s>"
unk_word = "<unk>"
pad_id = 0
bos_id = 1
eos_id = 2
unk_id = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_sentence(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def load_from_pickle(pickle_file):
	with open(pickle_file, "rb") as pickle_in:
		return pickle.load(pickle_in)

# Loading the pre-processed conversational exchanges (source-target pairs) from pickle data files
all_conversations = load_from_pickle("processed_CMDC.pkl")

# Extract 100 conversations from the end for evaluation and keep the rest for training
eval_conversations = all_conversations[-100:]
all_conversations = all_conversations[:-100]

class Vocabulary:
    def __init__(self):
        self.word_to_id = {pad_word: pad_id, bos_word: bos_id, eos_word:eos_id, unk_word: unk_id}
        self.word_count = {}
        self.id_to_word = {pad_id: pad_word, bos_id: bos_word, eos_id: eos_word, unk_id: unk_word}
        self.num_words = 4

    def get_ids_from_sentence(self, sentence):
        sentence = normalize_sentence(sentence)
        sent_ids = [bos_id] + [self.word_to_id[word] if word in self.word_to_id \
                               else unk_id for word in sentence.split()] + \
                               [eos_id]
        return sent_ids

    def tokenized_sentence(self, sentence):
        sent_ids = self.get_ids_from_sentence(sentence)
        return [self.id_to_word[word_id] for word_id in sent_ids]

    def decode_sentence_from_ids(self, sent_ids):
        words = list()
        for i, word_id in enumerate(sent_ids):
            if word_id in [bos_id, eos_id, pad_id]:
                # Skip these words
                continue
            else:
                # Add this error handling
                if word_id not in self.id_to_word:
                    print(f"Warning: Unknown token ID {word_id}, replacing with <unk>")
                    words.append(unk_word)
                else:
                    words.append(self.id_to_word[word_id])
        return ' '.join(words)

    def add_words_from_sentence(self, sentence):
        sentence = normalize_sentence(sentence)
        for word in sentence.split():
            if word not in self.word_to_id:
                # add this word to the vocabulary
                self.word_to_id[word] = self.num_words
                self.id_to_word[self.num_words] = word
                self.word_count[word] = 1
                self.num_words += 1
            else:
                # update the word count
                self.word_count[word] += 1
                
vocab = Vocabulary()
for src, tgt in all_conversations[:-100]:  # Skip the last 100 for eval
    vocab.add_words_from_sentence(src)
    vocab.add_words_from_sentence(tgt)
                
class Seq2seqBaseline(nn.Module):
    def __init__(self, vocab, emb_dim = 300, hidden_dim = 300, num_layers = 2, dropout=0.1):
        super().__init__()

        self.num_words = num_words = vocab.num_words
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Shared embedding layer for encoder and decoder
        self.embedding_layer = nn.Embedding(num_words, emb_dim)

        # Bidirectional GRU encoder
        self.encoder_gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Unidirectional GRU decoder
        self.decoder_gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0
        )

        # Linear layer to project decoder output to vocabulary
        self.output_projection = nn.Linear(hidden_dim, num_words)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Device for tensor allocation
        self.device = device

    def encode(self, source):
        # Compute a tensor containing the length of each source sequence.
        source_lengths = torch.sum(source != pad_id, axis=0).cpu()

        # Compute the mask first
        mask = (source == pad_id)

        # Convert word indexes to embeddings
        embedded = self.dropout(self.embedding_layer(source))

        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, source_lengths)

        # Forward pass through GRU
        outputs, hidden = self.encoder_gru(packed)

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        batch_size = source.size(1)
        outputs = outputs[:, :, :self.hidden_dim] + outputs[:, :, self.hidden_dim:]

        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        hidden = hidden.sum(dim=1)

        return outputs, mask, hidden

    def decode(self, decoder_input, last_hidden, encoder_output, encoder_mask):
        # These arguments are not used in the baseline model.
        del encoder_output
        del encoder_mask

        output, hidden = None, None
        
        # First process the decoder_input via embedding layer
        embedded = self.dropout(self.embedding_layer(decoder_input))

        # Forward through unidirectional GRU
        output, hidden = self.decoder_gru(embedded, last_hidden)
        output = self.output_projection(output.squeeze(0))

        return output, hidden, None

    def compute_loss(self, source, target):
        loss = 0
        
        # Forward pass through encoder
        encoder_outputs, encoder_mask, encoder_hidden = self.encode(source)

        # Create initial decoder input (start with SOS tokens for each sentence)
        batch_size = source.size(1)
        decoder_input = torch.full((1, batch_size), bos_id, dtype=torch.long, device=source.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden

        # Forward batch of sequences through decoder one time step at a time
        target_length = target.size(0)
        total_loss = 0
        total_tokens = 0

        for t in range(target_length):
            # Get decoder output for current time step
            decoder_output, decoder_hidden, _ = self.decode(
                decoder_input, decoder_hidden, encoder_outputs, encoder_mask)

            # Teacher forcing: next input is current target
            decoder_input = target[t:t+1]

            # Calculate and accumulate loss
            # Only calculate loss for non-padding tokens
            non_pad_mask = (target[t] != pad_id)
            num_valid_tokens = non_pad_mask.sum().item()

            if num_valid_tokens > 0:
                # Compute cross entropy loss for valid tokens
                loss = F.cross_entropy(
                    decoder_output[non_pad_mask],
                    target[t][non_pad_mask],
                    reduction='sum'
                )
                total_loss += loss
                total_tokens += num_valid_tokens

        # Return average loss over all non-padding tokens
        if total_tokens > 0:
            loss = total_loss / total_tokens
        else:
            loss = total_loss

        return loss
    
class Seq2seqAttention(Seq2seqBaseline):
    def __init__(self, vocab):
        super().__init__(vocab)

        self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_encoder = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def decode(self, decoder_input, last_hidden, encoder_output, encoder_mask):
        output, hidden, attn_weights = None, None, None

        # Get embedding of current input word
        embedded = self.dropout(self.embedding_layer(decoder_input))

        # Forward through unidirectional GRU
        output, hidden = self.decoder_gru(embedded, last_hidden)

        # Calculate attention weights from the current GRU output
        # encoder_output = self.attn(encoder_output)
        query = self.attn(output.transpose(0, 1))
        keys = encoder_output.transpose(0, 1)
        attn_scores = torch.bmm(query, keys.transpose(1, 2))

        if encoder_mask is not None:
            mask = encoder_mask.transpose(0, 1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=2)

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = torch.bmm(attn_weights, keys)

        # Concatenate weighted context vector and GRU output
        output = output.transpose(0, 1).squeeze(1)
        context = context.squeeze(1)

        combined = torch.cat((output, context), dim=1)
        combined = self.attn_combine(combined)
        combined = F.relu(combined)

        output = self.output_projection(combined)

        attn_weights = attn_weights.squeeze(1)

        return output, hidden, attn_weights
    
def predict_greedy(model, sentence, max_length=100):
    model.eval()

    generation = None

    # Forward input through encoder model
    with torch.no_grad():
        input_tensor = torch.tensor([vocab.get_ids_from_sentence(sentence)]).t().to(device)
        encoder_outputs, encoder_mask, encoder_hidden = model.encode(input_tensor)

    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden

    # Initialize decoder input with SOS_token
    decoder_input = torch.tensor([[bos_id]]).to(device)

    # Initialize tensors to append decoded words to
    decoded_tokens = []

    # Iteratively decode one word token at a time
    for _ in range(max_length):
        # Forward pass through decoder
        decoder_output, decoder_hidden, _ = model.decode(decoder_input, decoder_hidden, encoder_outputs, encoder_mask)
        # Obtain most likely word token and its softmax score
        topv, topi = decoder_output.topk(1)
        token = topi.item()

        # Record token and score
        decoded_tokens.append(token)

        if token == eos_id:
            break

        # Prepare current token to be next decoder input (add a dimension)
        decoder_input = torch.tensor([[token]]).to(device)

    # Return collections of word tokens and scores
    generation = vocab.decode_sentence_from_ids(decoded_tokens)

    return generation

def predict_top_p(model, sentence, temperature=0.95, top_p=0.92, min_length=8, max_length=100):
    model.eval()

    # Forward input through encoder model
    with torch.no_grad():
        input_tensor = torch.tensor([vocab.get_ids_from_sentence(sentence)]).t().to(device)
        encoder_outputs, encoder_mask, encoder_hidden = model.encode(input_tensor)

    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden
    # Initialize decoder input with SOS_token
    decoder_input = torch.tensor([[bos_id]]).to(device)

    # Initialize tensors to append decoded words to
    decoded_tokens = []
    
    # Get the vocabulary size to constrain output
    vocab_size = model.num_words

    # Iteratively decode one word token at a time
    for _ in range(max_length):
        # Forward pass through decoder
        decoder_output, decoder_hidden, _ = model.decode(decoder_input, decoder_hidden, encoder_outputs, encoder_mask)
        
        # Apply temperature scaling to the logits
        scaled_logits = decoder_output / temperature

        # Limit logits to valid vocabulary range
        if scaled_logits.size(1) > vocab_size:
            scaled_logits[:, vocab_size:] = float('-inf')

        # Sort logits in descending order (most to least probable)
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)

        # Calculate the cumulative sum of the token probabilities
        sorted_probs = F.softmax(sorted_logits, dim=1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=1)

        # Find the index where cumulative probability crosses top-p threshold
        indices_to_remove = cumulative_probs > top_p

        # Set the probabilities of all tokens after the top-p threshold to -inf
        indices_to_remove = torch.cat((torch.zeros(indices_to_remove.shape[0], 1).bool().to(device),
                                     indices_to_remove[:, :-1]), dim=1)
        sorted_logits[indices_to_remove] = float('-inf')

        # Re-compute the softmax of token probabilities and sample from the remaining logits
        filtered_probs = F.softmax(sorted_logits, dim=1)

        if (filtered_probs > 0).sum().item() > 0:
            selected_idx = torch.multinomial(filtered_probs, 1)
            token = sorted_indices.gather(1, selected_idx).item()
        else:
            token = decoder_output.argmax(1).item()

        # Skip EOS token if minimum length not reached
        if token == eos_id and len(decoded_tokens) < min_length:
            # Create a mask for the EOS token
            eos_mask = (sorted_indices[0] == eos_id)
            
            # Apply the mask correctly
            if eos_mask.any():
                # Zero out the probability for EOS token
                filtered_probs_copy = filtered_probs.clone()
                filtered_probs_copy[0, eos_mask] = 0
                
                # Check if there are any non-zero probabilities left
                if (filtered_probs_copy > 0).sum().item() > 0:
                    selected_idx = torch.multinomial(filtered_probs_copy[0], 1).unsqueeze(0)
                    token = sorted_indices[0, selected_idx[0, 0]].item()
                else:
                    # If no candidates remain, get the second-best token
                    token = decoder_output.topk(2)[1][0, 1].item()
            else:
                # If EOS is not in the top-p tokens, just continue
                pass

        # Double-check the token is within vocabulary range
        if token >= vocab_size:
            print(f"Warning: Generated token ID {token} is outside vocabulary range, using <unk> instead")
            token = unk_id

        # Record token
        decoded_tokens.append(token)

        if token == eos_id:
            break

        # Prepare current token to be next decoder input (add a dimension)
        decoder_input = torch.tensor([[token]]).to(device)

    # Return the decoded sentence
    return vocab.decode_sentence_from_ids(decoded_tokens)
    
def predict_beam(model, sentence, k=5, max_length=100):
    alpha = 0.7
    model.eval()

    generation = None

    with torch.no_grad():
        input_tensor = torch.tensor([vocab.get_ids_from_sentence(sentence)]).t().to(device)
        encoder_outputs, encoder_mask, encoder_hidden = model.encode(input_tensor)

        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[bos_id]]).to(device)

        beam = [(
            [bos_id],
            0.0,
            decoder_hidden)]

        complete_sequences = []

        for t in range(max_length):
            if len(complete_sequences) >= k:
                break

            all_candidates = []

            for seq, score, hidden in beam:
                if seq[-1] == eos_id:
                    normalized_score = score / (len(seq) ** alpha)
                    complete_sequences.append((seq, normalized_score))
                    continue

                decoder_input = torch.tensor([[seq[-1]]]).to(device)
                output, new_hidden, _ = model.decode(decoder_input, hidden, encoder_outputs, encoder_mask)

                log_probs = F.log_softmax(output, dim=1)

                topk = k if len(seq) == 1 else 2*k
                topk_log_probs, topk_indices = log_probs.topk(topk)

                for i in range(topk):
                    new_seq = seq + [topk_indices[0, i].item()]
                    new_score = score + topk_log_probs[0, i].item()
                    all_candidates.append((new_seq, new_score, new_hidden))

            if not all_candidates:
                break

            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = all_candidates[:k]

        while beam and len(complete_sequences) < k:
            seq, score, _ = beam.pop(0)

            if seq[-1] != eos_id:
                seq = seq + [eos_id]
            normalized_score = score / (len(seq) ** alpha)
            complete_sequences.append((seq, normalized_score))

        complete_sequences.sort(key=lambda x: x[1], reverse=True)
        generation = [vocab.decode_sentence_from_ids(seq) for seq, _ in complete_sequences]

    return generation

def load_models(vocab_file, baseline_model_file, attention_model_file, device):
    # Load vocabulary
    with open(vocab_file, "rb") as f:
        all_conversations = pickle.load(f)
    
    vocab = Vocabulary()
    for src, tgt in all_conversations[:-100]:  # Skip the last 100 for eval
        vocab.add_words_from_sentence(src)
        vocab.add_words_from_sentence(tgt)
    
    # Load models
    baseline_model = Seq2seqBaseline(vocab).to(device)
    baseline_model.load_state_dict(torch.load(baseline_model_file, map_location=device))
    baseline_model.eval()
    
    attention_model = Seq2seqAttention(vocab).to(device)
    attention_model.load_state_dict(torch.load(attention_model_file, map_location=device))
    attention_model.eval()
    
    return vocab, baseline_model, attention_model