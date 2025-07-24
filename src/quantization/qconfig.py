from typing import Any

def prepare_quantization_config(group_size: int, format: str) -> dict[str, Any]:
    if format == "mxfp":
        return {
            "forward_dtype": "mxfp4",
            "backward_dtype": "bf16",
            "forward_method": "abs_max",
            "hadamard_group_size": group_size,
            "modules_to_not_convert": ["lm_head"],
            "quant_method": "fp_quant",
            "store_master_weights": False
        }
    elif format == "nvfp":
        raise NotImplementedError("nvfp format is not supported yet")
    else:
        raise ValueError(f"Invalid format: {format}")
