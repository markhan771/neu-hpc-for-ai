from .sampling import SamplingParams
from .sequence import Sequence, SequenceStatus
from .engine import Engine


class Scheduler:
    def __init__(self, model_path: str, max_batch_size: int = 64,
                 device: str = "cuda"):
        self.engine = Engine(model_path, device=device)
        self.tokenizer = self.engine.tokenizer
        self.max_batch_size = max_batch_size

        self.next_seq_id = 0
        self.waiting_queue: list[Sequence] = []
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []

    def add_request(self, prompt: str,
                    sampling_params: SamplingParams = None):
        """Tokenize prompt, create Sequence, add to waiting queue."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(
            seq_id=self.next_seq_id,
            prompt_token_ids=token_ids,
            max_tokens=sampling_params.max_tokens,
        )
        self.waiting_queue.append(seq)
        self.next_seq_id += 1

    def _prefill_waiting(self, sampling_params: SamplingParams):
        """Prefill one request from the waiting queue and move to running."""
        if not self.waiting_queue:
            return
        if len(self.running) >= self.max_batch_size:
            return
        seq = self.waiting_queue.pop(0)
        first_token = self.engine.prefill(seq, sampling_params)
        seq.output_token_ids.append(first_token)
        if first_token == self.tokenizer.eos_token_id:
            seq.status = SequenceStatus.FINISHED
            self.finished.append(seq)
        else:
            self.running.append(seq)

    def _decode_running(self, sampling_params: SamplingParams):
        """Decode all running sequences in one batched forward pass.
        
        Steps:
          1. Call engine.decode_batch() with all running sequences
          2. Append each new token to its sequence
          3. Move finished sequences (EOS or max_tokens) to self.finished
          4. Keep remaining sequences in self.running
        """
        if not self.running:
            return

        # One batched GPU call for all running sequences
        next_tokens = self.engine.decode_batch(self.running, sampling_params)

        still_running = []
        for seq, token in zip(self.running, next_tokens):
            seq.output_token_ids.append(token)

            # Check stopping conditions
            is_eos       = (token == self.tokenizer.eos_token_id)
            is_max_tokens = (len(seq.output_token_ids) >= seq.max_tokens)

            if is_eos or is_max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.finished.append(seq)
            else:
                still_running.append(seq)

        self.running = still_running

    def step(self, sampling_params: SamplingParams = None):
        """One scheduling iteration.
        
        Order:
          1. Prefill one waiting request (if any, and batch not full)
          2. Decode all currently running requests in one batch
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Prefill one waiting request per step
        self._prefill_waiting(sampling_params)

        # Batch decode all running sequences
        self._decode_running(sampling_params)

    def run_to_completion(self,
                          sampling_params: SamplingParams = None) -> list[str]:
        """Run all requests to completion, return generated texts in order.
        
        Steps:
          1. Keep calling step() until waiting and running are both empty
          2. Sort finished sequences by seq_id to preserve original order
          3. Decode token IDs to strings, strip EOS token if present
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Run until all requests are finished
        while self.waiting_queue or self.running:
            self.step(sampling_params)

        # Sort by seq_id to return results in submission order
        self.finished.sort(key=lambda s: s.seq_id)

        results = []
        for seq in self.finished:
            ids = seq.output_token_ids
            # Strip trailing EOS if present
            if ids and ids[-1] == self.tokenizer.eos_token_id:
                ids = ids[:-1]
            results.append(self.tokenizer.decode(ids))

        return results