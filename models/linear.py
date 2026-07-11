from torch import nn

class Linear(nn.Module):
    """Flatten (B, T, F) -> logits. 의도적으로 약한 baseline: temporal structure를 명시적으로 모델링하지 X!"""
    def __init__(self, d_inp, max_len, n_classes):
        super().__init__()
        self.d_inp = d_inp
        self.max_len = max_len
        self.n_classes = n_classes
        self.clf = nn.Linear(d_inp * max_len, n_classes)

    def forward(self, x, mask=None, timesteps=None, get_embedding=False,
                captum_input=False, show_sizes=False, return_all=False):
        # x: (B, T, F)
        B = x.shape[0]
        emb = x.reshape(B, -1)      # (B, T*F)
        out = self.clf(emb)          # (B, n_classes)
        if get_embedding:
            return out, emb
        return out
