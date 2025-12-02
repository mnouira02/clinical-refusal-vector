import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class VeritasEngine:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.activations = {}
        self.hook_handles = []
        
        # CONFIGURATION: 4-bit Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",            
            bnb_4bit_compute_dtype=torch.float16, 
            bnb_4bit_use_double_quant=True,
        )

        print(f"Loading {model_id} into Research Engine (4-bit)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully.")

    def _get_activation_hook(self, name):
        """Captures hidden states, handling both Tuple and Tensor outputs."""
        def hook(module, input, output):
            # Check if output is a tuple (Standard) or Tensor (Edge case)
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook

    def extract_vector(self, positive_prompt, negative_prompt, layer_id):
        """Calculates the steering vector."""
        layer = self.model.model.layers[layer_id]
        handle = layer.register_forward_hook(self._get_activation_hook("target"))
        
        try:
            # 1. Positive Pass
            inputs_pos = self.tokenizer(positive_prompt, return_tensors="pt").to(self.device)
            self.model(**inputs_pos)
            
            # ROBUST EXTRACT: Handle missing batch dimension
            act_pos = self.activations["target"]
            if len(act_pos.shape) == 3:
                pos_vec = act_pos[0, -1, :]
            else:
                pos_vec = act_pos[-1, :]

            # 2. Negative Pass
            inputs_neg = self.tokenizer(negative_prompt, return_tensors="pt").to(self.device)
            self.model(**inputs_neg)
            
            act_neg = self.activations["target"]
            if len(act_neg.shape) == 3:
                neg_vec = act_neg[0, -1, :] 
            else:
                neg_vec = act_neg[-1, :]

            # 3. Calculate
            steering_vec = pos_vec - neg_vec
            
            # Cleanup
            del inputs_pos, inputs_neg, pos_vec, neg_vec, act_pos, act_neg
            torch.cuda.empty_cache()
            
            return steering_vec
            
        finally:
            handle.remove()

    def apply_steering(self, vector, layer_id, coeff=1.0):
        """Injects the Magnet."""
        layer = self.model.model.layers[layer_id]
        
        def steering_hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            if is_tuple:
                h_state = output[0]
            else:
                h_state = output

            dtype = h_state.dtype
            device = h_state.device
            v = vector.to(device).to(dtype)
            h_state[:, :, :] += coeff * v
            
            if is_tuple:
                return (h_state,) + output[1:]
            else:
                return h_state

        handle = layer.register_forward_hook(steering_hook)
        self.hook_handles.append(handle)

    def reset_steering(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []