# PicoLM-Zig Ultra
Run a 1-billion parameter LLM on consumer hardware with zero dependencies. Pure Zig. High performance. Cross-platform.

```powershell
echo "Explain gravity" | .\picolm.exe model.gguf --q8k -n 100 -j 8
```

PicoLM-Zig Ultra is a high-performance, minimalist inference engine for GGUF models, migrated to **Zig 0.13.0**. It delivers "Zigsteroids" performance with native support for Windows and Linux.

## 🚀 Key Features

- **⚡ Blazing Fast**: Heavy use of SIMD (AVX2, FMA) and hardware-specific optimizations.
- **✨ Pure Zig**: Minimal codebase, zero dependencies, lightning-fast compilation.
- **🪟 Windows & Linux**: Native support with high-performance memory mapping fallbacks for Windows.
- **🔢 Multi-Quant Support**: Full support for `Q2_K`, `Q4_K`, `Q8_0`, and `Q8_K` quantization modes.
- **🧠 Hardcore Optimizations**:
  - **CCX Pinning**: Optimized for Zen 2/3 architectures.
  - **NUMA Interleaving**: Efficient memory access for dual-socket servers.
  - **Thermal Throttling**: Intelligent monitoring to prevent hardware overheating.
- **🛠️ Self-Contained**: No Python, no CUDA (unless explicitly enabled), just one binary.

## 🛠️ Installation & Build

Requires [Zig 0.13.0+](https://ziglang.org/download/).

### Build from source
```bash
# Clone the repository
git clone https://github.com/Cristian/picolm-on-zigsteroids.git
cd picolm-on-zigsteroids

# Build for your architecture (Optimized)
zig build -Doptimize=ReleaseFast
```

The binary will be available in `zig-out/bin/picolm`.

## 📖 Usage

```text
Usage: picolm <model.gguf> [options]

Options:
  -p <prompt>      Input prompt
  -n <int>         Max tokens (default: 256)
  -t <float>       Temperature (default: 0.8)
  -k <float>       Top-p (default: 0.9)
  -s <int>         Seed (default: 42)
  -j <int>         Threads (default: auto)
  -c <int>         Context length
  --q2k, --q4k, --q80, --q8k  Quantization selection
  --info           Show model info
  --memory         Show memory breakdown
  --ccx-optimize   Zen 2/3 CCX pinning (AMD Ryzen)
  --numa-interleave Dual-socket NUMA optimization
  --thermal-limit  Enable thermal monitoring
  -h, --help       Show help
```

## 📈 Performance Tip

For maximum performance on AMD Ryzen processors, use the `--ccx-optimize` flag to pin threads to a single Core Complex, reducing latency.

---
*PicoLM-Zig Ultra — Intelligence shouldn't require a data center.*
