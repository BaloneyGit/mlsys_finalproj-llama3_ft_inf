import math
import torch
import torch.nn as nn

class LoRALayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0
    ):
        self.r = r
        if r > 0:
            # A and B
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))

            # scaling B.(A(x))
            self.scaling = lora_alpha / r
            self.lora_dropout = nn.Dropout(lora_dropout)

            # initialize A and B
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def lora_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        calculating lora adapter output: LoRA_dropout(B.(A(x))) * scaling
        """
        Ax = x.transpose(-1, -2)
        Ax = self.lora_A @ Ax
        Ax = self.lora_dropout(Ax.transpose(-1, -2))
        BA = (Ax @ self.lora_B.transpose(0, 1))
        return BA.view(*x.size()[:-1], -1) * self.scaling

    
class Linear(nn.Linear, LoRALayer):
    """
    forward pass: W_0(x) + LoRA_dropout(B.(A(x))) * scaling
    """
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        bias: bool = True
    ):
        # initialize frozen weight and bias
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        # initialize LoRA adapter parameters
        LoRALayer.__init__(self, in_features, out_features, r, lora_alpha, lora_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base projection
        result = nn.Linear.forward(self, x)

        # adding LoRA if enabled
        if self.r > 0:
            result = result + self.lora_forward(x)
        return result
