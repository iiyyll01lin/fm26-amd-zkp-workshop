#!/usr/bin/env bash
# Where does the risc0 ROCm prover actually spend its time, and does moving
# witgen/accum onto the GPU help?
#
# The hybrid gate leaves witness generation and accumulation on the CPU by
# default; both have real HIP implementations behind RISC0_ROCM_WITGEN and
# RISC0_ROCM_ACCUM. Nobody had measured what turning them on costs or saves, so
# the Amdahl ceiling of the whole port was unknown.
#
# Every receipt is verified with stock cargo-risczero. The container timing
# harness cannot do this (its image ships a proving key but no verifying key),
# so this is also the first end-to-end timing evidence on this branch that is
# correctness-backed rather than timing-only.
set -uo pipefail

REPO="${HOME}/yy/workspace/zkp-final"
POC="${REPO}/poc/risc0-rocm-prover"
DEMO="${REPO}/poc/risc0-cartesi-step-demo"
R0VM="${R0VM:-/tmp/r0vm-target/release/r0vm}"
ARCH="${RISC0_HIP_OFFLOAD_ARCH:-gfx1201}"
REPS="${WITGEN_REPS:-5}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${WITGEN_OUT:-${POC}/artefacts/witgen-gate-${ARCH}-${STAMP}}"

export PATH="${HOME}/.cargo/bin:${HOME}/.local/hipcc-shim:${HOME}/.local/bin:/opt/rocm/bin:${PATH}"

# shellcheck disable=SC1091
source "${POC}/scripts/gpu-select.sh"
BDF="$(zkp_gpu_normalize_bdf "${ZKP_TARGET_GPU_BDF:-0000:03:00.0}")" || exit 2
zkp_gpu_require_idle "${BDF}" || { echo "target card not idle" >&2; exit 43; }
zkp_gpu_isolate "${BDF}" || exit 2
SMI="$(zkp_gpu_smi_index "${BDF}")"

# The export above must reach the prover; assert it rather than assume it.
mapfile -t BDFIDS < <(rocminfo 2>/dev/null | awk '/^ *BDFID:/{print $2}' | grep -v '^0$')
(( ${#BDFIDS[@]} == 1 )) || { echo "runtime sees ${#BDFIDS[@]} GPUs, expected 1" >&2; exit 1; }
(( (BDFIDS[0] >> 8) == 16#$(printf '%s' "${BDF}" | cut -d: -f2) )) || { echo "wrong GPU" >&2; exit 1; }

mkdir -p "${OUT}"
ELF="${DEMO}/dist/cartesi-risc0-guest-step-prover.bin"
INPUT="${DEMO}/artefacts/step.bin"
IMAGE_ID="$(tr -d '[:space:]' <"${DEMO}/dist/cartesi-risc0-guest-step-prover-image-id.txt")"

{
    echo "started_utc=$(date -u +%FT%TZ)"
    echo "arch=${ARCH}"
    echo "target_gpu_bdf=${BDF}"
    echo "rocr_index=${ZKP_SELECTED_ROCR_INDEX}"
    echo "rocm_smi_index=${SMI}"
    echo "runtime_visible_gpus=${#BDFIDS[@]}"
    echo "image_id=${IMAGE_ID}"
    echo "r0vm_sha256=$(sha256sum "${R0VM}" | awk '{print $1}')"
    echo "elf_sha256=$(sha256sum "${ELF}" | awk '{print $1}')"
    echo "input_sha256=$(sha256sum "${INPUT}" | awk '{print $1}')"
    echo "rayon_num_threads=${RAYON_NUM_THREADS:-32}"
    echo "reps=${REPS}"
    echo "host_nproc=$(nproc)"
    echo "cargo_risczero=$(cargo risczero --version 2>/dev/null | head -1)"
} >"${OUT}/provenance.txt"

CSV="${OUT}/witgen-gate.csv"
echo 'witgen,accum,rep,wall_s,energy_j,mean_power_w,gpu_busy_max,verify,receipt_sha256' >"${CSV}"

# amdgpu power1_average on this part is socket-wide, so both arms sit on the
# same basis; the absolute value is not a GPU-only figure.
HWMON="$(ls -d /sys/class/drm/card${SMI}/device/hwmon/hwmon*/ 2>/dev/null | head -1)"
BUSY="/sys/class/drm/card${SMI}/device/gpu_busy_percent"

sample_loop() {
    local out="$1"
    : >"${out}"
    while :; do
        printf '%s %s %s\n' "$(date +%s.%N)" \
            "$(cat "${HWMON}power1_average" 2>/dev/null || echo 0)" \
            "$(cat "${BUSY}" 2>/dev/null || echo 0)" >>"${out}"
        sleep 0.11
    done
}

for wg in 0 1; do
    for ac in 0 1; do
        for ((rep = 0; rep <= REPS; rep++)); do
            tag="w${wg}a${ac}-${rep}"
            receipt="${OUT}/receipt-${tag}.bin"
            samples="${OUT}/power-${tag}.txt"
            sample_loop "${samples}" &
            spid=$!
            start="$(date +%s.%N)"
            env RISC0_ROCM_WITGEN="${wg}" RISC0_ROCM_ACCUM="${ac}" \
                RISC0_HIP_OFFLOAD_ARCH="${ARCH}" \
                RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-32}" \
                "${R0VM}" --elf "${ELF}" --initial-input "${INPUT}" \
                --receipt "${receipt}" >"${OUT}/prove-${tag}.log" 2>&1
            rc=$?
            end="$(date +%s.%N)"
            kill "${spid}" 2>/dev/null
            wait "${spid}" 2>/dev/null

            wall="$(awk -v a="${start}" -v b="${end}" 'BEGIN{printf "%.3f", b-a}')"
            read -r energy power busymax < <(awk '
                {p=$2/1e6; b=$3+0
                 if (NR>1) {e += p*($1-t)}
                 t=$1; s+=p; n++; if (b>bm) bm=b}
                END{printf "%.1f %.2f %d", e, (n?s/n:0), bm}' "${samples}")

            if [[ ${rc} -ne 0 ]]; then
                verify="prove-failed"
                sha="-"
            else
                sha="$(sha256sum "${receipt}" | awk '{print $1}')"
                if cargo risczero verify --path "${receipt}" "${IMAGE_ID}" \
                    >"${OUT}/verify-${tag}.log" 2>&1; then
                    verify="PASS"
                else
                    verify="FAIL"
                fi
            fi
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "${wg}" "${ac}" "${rep}" "${wall}" "${energy}" "${power}" \
                "${busymax}" "${verify}" "${sha}" >>"${CSV}"
            echo "witgen=${wg} accum=${ac} rep=${rep} wall=${wall}s energy=${energy}J verify=${verify}"
            rm -f "${receipt}"
            sleep 3
        done
    done
done

python3 - "${CSV}" "${OUT}/summary.txt" <<'PY'
import csv, statistics, sys

rows = list(csv.DictReader(open(sys.argv[1])))
out = []
by = {}
for r in rows:
    if r["rep"] == "0":            # discard the warm-up rep
        continue
    by.setdefault((r["witgen"], r["accum"]), []).append(r)

out.append(f"{'witgen':>7}{'accum':>7}{'n':>4}{'median_s':>11}{'min':>9}{'max':>9}"
           f"{'median_J':>11}{'gpu_busy_max':>14}{'verify':>10}")
base = None
for key in sorted(by):
    rs = by[key]
    walls = sorted(float(r["wall_s"]) for r in rs)
    energy = sorted(float(r["energy_j"]) for r in rs)
    med = statistics.median(walls)
    if key == ("0", "0"):
        base = med
    verdicts = {r["verify"] for r in rs}
    verdict = "PASS" if verdicts == {"PASS"} else "/".join(sorted(verdicts))
    out.append(f"{key[0]:>7}{key[1]:>7}{len(rs):>4}{med:>11.3f}{walls[0]:>9.3f}"
               f"{walls[-1]:>9.3f}{statistics.median(energy):>11.1f}"
               f"{max(int(r['gpu_busy_max']) for r in rs):>14}{verdict:>10}")

if base:
    out.append("")
    out.append("relative to witgen=CPU accum=CPU (the shipped default):")
    for key in sorted(by):
        med = statistics.median(float(r["wall_s"]) for r in by[key])
        out.append(f"  witgen={key[0]} accum={key[1]}  {base / med:.3f}x")

text = "\n".join(out)
open(sys.argv[2], "w").write(text + "\n")
print(text)
PY

( cd "${OUT}" && find . -type f ! -name evidence.sha256 -print0 | sort -z |
  xargs -0 sha256sum >evidence.sha256 )
echo "witgen gate sweep complete: ${OUT}"
