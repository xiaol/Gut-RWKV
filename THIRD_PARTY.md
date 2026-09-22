# Third-party sources

`rwkv_jev/backbone/rwkv7.py`, `wkv7_kernel.py`, and `cuda/wkv7_state.*`
are vendored from [xiaol/State-Bridge](https://github.com/xiaol/State-Bridge)
at commit `0b4eb711ed1c02ca05d96a2e7fbdfba1d438a174`. They implement the RWKV-7
reference equations with explicit recurrent state and an initial-state-gradient
CUDA kernel. The original MIT license is retained at
`rwkv_jev/backbone/LICENSE`. See also the official
[RWKV-LM implementation](https://github.com/BlinkDL/RWKV-LM).

The vendored kernel does not differentiate its final state output; Gut-RWKV
therefore trains on complete candidate sequences. It does differentiate the
initial state input, which enables initial WKV state tuning.

Base weights and the World tokenizer are downloaded separately from BlinkDL;
they are not relicensed by this project. Kev benchmark data retains each source
dataset's license. No upstream training records or model weights are included
in the source repository. NanoJev, Kev and Jev are cited as prior work; Gut-RWKV
does not contain their model implementation code.
