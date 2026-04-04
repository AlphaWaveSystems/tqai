# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in tqai, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Use [GitHub's private vulnerability reporting](https://github.com/AlphaWaveSystems/tqai/security/advisories/new)
3. Or email **github@alphawavesystems.com**

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Scope

tqai is a quantization library that processes model weights and KV cache tensors locally. Key security considerations:

- **Model loading**: tqai loads models via HuggingFace or mlx-lm. Ensure you trust the model source.
- **Serialized data**: `tqai convert` saves `.npz` files. Only load converted configs from trusted sources.
- **No network access**: tqai itself does not make network requests (model downloading is handled by upstream libraries).

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
