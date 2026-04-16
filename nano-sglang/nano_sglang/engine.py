import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from .model import Model, Tokenizer
from .sampling import SamplingParams, sample_token
from .sequence import Sequence, SequenceStatus


class Engine:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = Model(model_path, device=device)
        self.tokenizer = Tokenizer(model_path)
        self.device = device

    def prefill(self, seq: Sequence, sampling_params: SamplingParams) -> int:
        """Process all prompt tokens in one forward pass, return first generated token.
        
        Steps:
          1. Convert prompt token IDs to tensor
          2. Run one forward pass through the model (no past KV yet)
          3. Store the returned past_key_values in seq
          4. Sample the first output token from the last logit position
          5. Set seq.status to DECODING
        """
        input_ids = torch.tensor(
            [seq.prompt_token_ids], device=self.device
        )  # [1, prompt_len]

        # Forward pass — no past KV cache yet
        logits, past_key_values = self.model.forward(
            input_ids, past_key_values=None
        )

        # Store KV cache in the sequence for future decode steps
        seq.past_key_values = past_key_values

        # Sample the first output token from the last position
        next_token = sample_token(logits[:, -1, :], sampling_params).item()

        # Mark sequence as decoding
        seq.status = SequenceStatus.DECODING

        return next_token

    def decode_step(self, seq: Sequence, sampling_params: SamplingParams) -> int:
        """Generate one token for a single sequence using cached KV."""
        last_token = seq.output_token_ids[-1]
        input_ids = torch.tensor([[last_token]], device=self.device)
        logits, past_key_values = self.model.forward(
            input_ids, past_key_values=seq.past_key_values)
        next_token = sample_token(logits[:, -1, :], sampling_params).item()
        seq.past_key_values = past_key_values
        return next_token

    def decode_batch(self, sequences: list[Sequence],
                     sampling_params: SamplingParams) -> list[int]:
        """Generate one token for multiple sequences in a single GPU forward pass."""
        if not sequences:
            return []
        if len(sequences) == 1:
            return [self.decode_step(sequences[0], sampling_params)]

        n = len(sequences)
        input_ids = torch.tensor(
            [[seq.output_token_ids[-1]] for seq in sequences],
            device=self.device,
        )

        cache_lens = [seq.past_key_values.get_seq_length() for seq in sequences]
        max_len = max(cache_lens)

        batched_cache = DynamicCache()
        for layer_idx in range(self.model.num_layers):
            padded_keys, padded_values = [], []
            for seq in sequences:
                k = seq.past_key_values.key_cache[layer_idx]
                v = seq.past_key_values.value_cache[layer_idx]
                pad = max_len - k.shape[2]
                if pad > 0:
                    k = F.pad(k, (0, 0, pad, 0))
                    v = F.pad(v, (0, 0, pad, 0))
                padded_keys.append(k)
                padded_values.append(v)
            batched_cache.key_cache.append(torch.cat(padded_keys, dim=0))
            batched_cache.value_cache.append(torch.cat(padded_values, dim=0))

        attn_mask = torch.zeros(n, max_len + 1, device=self.device,
                                dtype=torch.long)
        for i, cl in enumerate(cache_lens):
            attn_mask[i, max_len - cl:] = 1

        position_ids = torch.tensor(
            [[cl] for cl in cache_lens], device=self.device)

        logits, new_cache = self.model.forward(
            input_ids, past_key_values=batched_cache,
            position_ids=position_ids, attention_mask=attn_mask,
        )

        tokens = sample_token(logits[:, -1, :], sampling_params)

        for i, seq in enumerate(sequences):
            real_len = cache_lens[i] + 1
            pad = max_len - cache_lens[i]
            per_seq_cache = DynamicCache()
            for layer_idx in range(self.model.num_layers):
                k = new_cache.key_cache[layer_idx][
                    i:i+1, :, pad:pad + real_len, :]
                v = new_cache.value_cache[layer_idx][
                    i:i+1, :, pad:pad + real_len, :]
                per_seq_cache.key_cache.append(k.clone())
                per_seq_cache.value_cache.append(v.clone())
            seq.past_key_values = per_seq_cache

        return [t.item() for t in tokens]

    def generate(self, prompt: str,
                 sampling_params: SamplingParams = None) -> str:
        """Generate text for a single prompt.
        
        Steps:
          1. Create a Sequence from the prompt
          2. Prefill → get first token
          3. Decode loop until EOS or max_tokens
          4. Decode output token IDs back to string
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Create sequence
        token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=token_ids,
            max_tokens=sampling_params.max_tokens,
        )

        # Prefill: process entire prompt, get first token
        first_token = self.prefill(seq, sampling_params)
        seq.output_token_ids.append(first_token)

        # Decode loop
        while True:
            last_token = seq.output_token_ids[-1]

            # Stop at EOS
            if last_token == self.tokenizer.eos_token_id:
                break

            # Stop at max_tokens
            if len(seq.output_token_ids) >= sampling_params.max_tokens:
                break

            # Generate next token
            next_token = self.decode_step(seq, sampling_params)
            seq.output_token_ids.append(next_token)

        # Decode token IDs to string (exclude EOS if present)
        output_ids = seq.output_token_ids
        if output_ids and output_ids[-1] == self.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]

        return self.tokenizer.decode(output_ids)