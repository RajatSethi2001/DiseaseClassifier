import numpy as np
import random
import selfies as sf
import torch

def one_hot_encode(tokens, alphabet):
    vocab_dict = {token: i for i, token in enumerate(alphabet)}
    indices = [vocab_dict[token] for token in tokens]
    one_hot = np.eye(len(vocab_dict))[indices]
    return one_hot

def one_hot_decode(one_hot, alphabet):
    tokens = []
    for vector in one_hot:
        index = np.argmax(vector)
        token = alphabet[index]
        tokens.append(token)
    return tokens

def clean_dose_unit(raw):
    if isinstance(raw, bytes):
        if raw in [b'\xfd\xfdM', b'\xc2\xb5M']:
            return 'uM'
        elif raw == b'\xfd\xfdL':
            return 'uL'
        elif raw == b'ng/mL':
            return 'ng/mL'
        elif raw == b'ng':
            return 'ng'
        elif raw == b'ng/\xfd\xfdL':
            return 'ng/uL'
        elif raw == b'-666':
            return None
        else:
            return raw.decode('utf-8', errors='ignore')  # fallback
    return raw

def get_minmax(vector):
    lo = np.min(vector)
    hi = np.max(vector)
    return np.clip((vector - lo) / (hi - lo), 0, 1)

def get_zscores(vector):
    return (vector - np.mean(vector)) / np.std(vector)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def smiles_to_embedding(smiles, selfies_alphabet, encoder, max_selfies_len=50):
    selfies = sf.encoder(smiles)
    selfies_tokens = list(sf.split_selfies(selfies))
    selfies_tokens += ["[SKIP]" for _ in range(max_selfies_len - len(selfies_tokens))]
    selfies_one_hot = torch.tensor(one_hot_encode(selfies_tokens, selfies_alphabet), dtype=torch.float32).unsqueeze(0)
    selfies_embedding = encoder(selfies_one_hot)[0]
    return selfies_embedding.detach().cpu().numpy()

def embedding_to_smiles(embedding, selfies_alphabet, decoder):
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
    selfies_one_hot = decoder(embedding_tensor)[0].detach().cpu().numpy()
    selfies_tokens = one_hot_decode(selfies_one_hot, selfies_alphabet)
    selfies_tokens_clean = [token for token in selfies_tokens if token != "[SKIP]"]
    selfies = "".join(selfies_tokens_clean)
    smiles = sf.decoder(selfies)
    return smiles