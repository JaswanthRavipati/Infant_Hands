print("🚀 Testing SAM2 load...")

SAM_CHECKPOINT = "/N/slate/veravi/model/sam2_hiera_large.pt"

from sam2.build_sam import build_sam2

CONFIG_ROOT = "/N/u/veravi/BigRed200/.conda/envs/hand_env/lib/python3.11/site-packages/sam2/configs"

try:
    sam_model = build_sam2(
        config_file="sam2_hiera_l",
        ckpt_path=SAM_CHECKPOINT,
        hydra_overrides_extra=[
            f'hydra.searchpath=[file://{CONFIG_ROOT}/sam2]'
        ]
    )
    print("✅ SAM2 loaded successfully!")

except Exception as e:
    print("❌ SAM2 failed to load")
    print(e)