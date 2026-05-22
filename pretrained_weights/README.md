# Pretrained Weights

This directory is intentionally released without model weight files.

Before running MQ-Auditor, prepare the following local layout:

```text
pretrained_weights/
  Llama-2-7b-chat-hf/
  clip-vit-large-patch14/
  google-bert-base-uncased/
  BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
  audio_pretrain.bin
  visual_pretrain.bin
```

The empty directories are kept as placeholders. Download the actual model files according to the upstream licenses and terms. For a related setup reference, see [TGS-Agent](https://github.com/jasongief/TGS-Agent).
