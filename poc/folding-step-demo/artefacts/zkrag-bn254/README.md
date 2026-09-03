# STEP 3 — zkRAG retrieval relation re-cast to BN254, verified on-chain

**Result: the zkRAG HNSW top-k / membership relation, re-cast from the
bellperson/BLS12-381 bench to an arkworks `ConstraintSynthesizer<ark_bn254::Fr>`
and proven through the Demo C vendored+patched `ark-groth16` (BN254) prover,
verifies NATIVELY and ON-CHAIN (anvil) via a snarkJS-style Groth16 Solidity
verifier.**

```
[zkrag-bn254] instance n=32 d=16 k=5 ef=20 M=8 | top_ids=[15,14,20,7,28] top_dists=[500,547,598,685,691] recall=5
[zkrag-bn254] real-relation constraints = 10874 (~2^13.4)
[zkrag-bn254] native Groth16 verify: OK  (13 public inputs)
  Groth16Verifier(zkRAG BN254 proof) -> true  [ZKRAG BN254 PROOF VERIFIED ON-CHAIN]
```

## Why this was the genuine STEP 3 work (the framework mismatch, resolved)

The runnable iGPU zkRAG bench (`poc/amd-gpu-zk-primitive-demo/src/bin/zkrag-retrieval-msm.rs`)
is a **`bellperson::Circuit` over BLS12-381** (`blstrs`). bellperson's OpenCL MSM is
built in (so the BLS12-381 iGPU path needs no fork), but bellperson is hardwired to
BLS12-381 — so that circuit **cannot** be fed to a BN254 Groth16 prover and therefore
cannot verify on the EVM (Ethereum's pairing precompiles are BN254). `docs/zkrag-igpu-proof-scope.md`
§4.3 / §6(4) recorded this as "blocked at the `ark-groth16` injection seam."

That blocker is **now resolved**: Demo C vendored + patched `ark-groth16` 0.5.0
(`poc/folding-step-demo/vendor/ark-groth16`) with a feature/env-gated GPU-MSM seam
(G1, G2(Fq2), QAP FFT). So the honest path is a **framework port**: re-express the
four checks + algebraic commitment as an **arkworks** R1CS over `ark_bn254::Fr` and
prove it with that vendored prover — the exact Demo C G1+G2+FFT offload path. The
bellperson circuit could not be "reused"; it had to be re-cast (different constraint
DSL). That port is `src/bin/zkrag-bn254-onchain.rs`.

## Faithfulness (relation parity across the port)

The arkworks re-cast recovers the **identical** brute-force top-k as the BLS12-381
bench — `top_ids=[15,14,20,7,28]`, `top_dists=[500,547,598,685,691]`, `recall=5`,
`~10.8k` constraints — over the same deterministic instance (n=32, d=16, k=5, ef=20,
M=8; squared-L2; LCG seed `0x5eed1234abcd0001`). It ports all four checks:
distance `Σ(v−q)²` + range; one-hot membership + ascending + separation top-k;
priority-queue monotone trace; adjacency lookup; plus the algebraic Horner commitment
`C = Σ_i w_i·g^i` (public `g`) — the MSM-shaped opening that feeds the prover G1/G2
MSM + QAP FFT the iGPU accelerates. A native `Groth16::verify` against
`[top_ids ‖ top_dists ‖ recall ‖ g ‖ C]` gates every run.

## CPU vs iGPU offload

The vendored prover is **byte-identical CPU** when the gpu-msm seam is dormant, and
Demo C proved GPU==CPU bit-for-bit for every primitive — so the **on-chain
verification is identical whether the proof was produced on CPU or iGPU**. The bin is
therefore fully runnable on CPU (default). The iGPU-offloaded *timing* variant is
opt-in (`FOLD_GPU_MSM=1 FOLD_GPU_G2=1 FOLD_GPU_FFT=1`, optionally
`ZKRAG_BN254_POW=22+`) and solo-guarded; its measurement was **blocked** this session
by a third-party ROCm `train_language.py` saturating the iGPU (see
`docs/INTEGRITY-REPORT.md` §6). Per the Demo C reversal, an iGPU offload at this size
is expected to be a *slowdown* anyway — the value here is the BN254 **on-chain
verification capability**, not a speed win.

## Reproduce

```bash
. "$HOME/.cargo/env"; export PATH="$HOME/.foundry/bin:$HOME/.risc0/bin:$PATH"
cd poc/folding-step-demo
bash scripts/run-zkrag-bn254-onchain.sh           # CPU prove + native + on-chain
# iGPU-offloaded prove at 2^22 (solo-gated; exits 42 under contention):
FOLD_GPU_MSM=1 FOLD_GPU_G2=1 FOLD_GPU_FFT=1 ZKRAG_BN254_POW=22 \
    bash scripts/run-zkrag-bn254-onchain.sh
```

Artefacts: `Groth16Verifier.sol` (the deployed verifier), `proof.json`
(a/b/c + 13 public inputs as decimal strings, consumed by
`forge/script/VerifyZkRag.s.sol`), `anvil.log`.
