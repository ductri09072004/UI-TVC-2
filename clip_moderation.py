"""
Module CLIP cho kiểm duyệt TVC.
Bao gồm pre-filtering và caption quality verification.
"""

import os
import torch
from typing import Dict, Optional, Tuple
from PIL import Image

# Try to import CLIP at module level so functions can use `clip.tokenize`
try:
    import clip  # type: ignore
except Exception:
    clip = None  # Will attempt to import inside loader as fallback

# Global CLIP model cache
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None

def _encode_text_labels(model, labels, device):
    """Encode labels with multilingual prompt templates and average their features.
    Returns a tensor of shape [len(labels), D].
    """
    assert clip is not None, "CLIP module not available"
    # Multilingual prompt templates to stabilize scores
    templates = [
        "a photo of {}",
        "an image of {}",
        "{}",
        "ảnh về {}",
        "hình ảnh về {}",
        "nội dung {}",
    ]
    all_feats = []
    with torch.no_grad():
        for label in labels:
            prompts = [t.format(label) for t in templates]
            text_tokens = clip.tokenize(prompts).to(device)
            text_features = model.encode_text(text_tokens)
            # Normalize then average
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_feat = text_features.mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm()
            all_feats.append(mean_feat)
        return torch.stack(all_feats, dim=0)


def _get_clip_model():
    """Load CLIP model (cached, singleton pattern)."""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    
    if _CLIP_MODEL is None:
        # Ensure clip is imported
        global clip
        if clip is None:
            try:
                import clip as _clip  # local import
                clip = _clip
            except ImportError as e:
                raise ImportError(
                    f"CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git\n"
                    f"Original error: {e}"
                )
        
        # Determine device
        _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model (ViT-B/32 is a good balance of speed and accuracy)
        model_name = os.environ.get("CLIP_MODEL", "ViT-B/32")
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(model_name, device=_CLIP_DEVICE)
        _CLIP_MODEL.eval()  # Set to evaluation mode
    
    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE


def pre_filter_image(
    image_path: str,
    violation_threshold: float = 0.3,
    skip_threshold: float = 0.7
) -> Dict[str, any]:
    """
    Pre-filter ảnh với CLIP để phát hiện vi phạm rõ ràng.
    
    Args:
        image_path: Đường dẫn đến ảnh
        violation_threshold: Threshold để coi là vi phạm (default: 0.3)
        skip_threshold: Threshold để skip BLIP + BERT (default: 0.7)
    
    Returns:
        Dict với:
            - is_violation: bool - Có phải vi phạm không
            - violation_score: float - Điểm vi phạm (0-1)
            - healthy_score: float - Điểm lành mạnh (0-1)
            - skip_processing: bool - Có nên skip BLIP + BERT không
            - method: str - "clip_prefilter"
    """
    if not os.path.exists(image_path):
        return {
            "is_violation": False,
            "violation_score": 0.0,
            "healthy_score": 1.0,
            "skip_processing": False,
            "method": "clip_prefilter",
            "error": "Image not found"
        }
    
    try:
        model, preprocess, device = _get_clip_model()
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Mô tả các loại vi phạm (zero-shot, có thể customize)
        violation_types = os.environ.get(
            "CLIP_VIOLATION_TYPES",
            "nội dung bạo lực|nội dung không phù hợp|nội dung nhạy cảm|nội dung quảng cáo sai sự thật"
        ).split("|")
        violation_types = [t.strip() for t in violation_types if t and t.strip()]
        
        healthy_type = os.environ.get("CLIP_HEALTHY_TYPE", "nội dung lành mạnh").strip()
        
        # Guard: if no violation types, never mark violation
        if len(violation_types) == 0:
            return {
                "is_violation": False,
                "violation_score": 0.0,
                "healthy_score": 1.0,
                "skip_processing": False,
                "method": "clip_prefilter",
                "violation_details": {}
            }

        # Build text features with multilingual templates (more robust than single tokenization)
        all_types = violation_types + [healthy_type]
        text_features = _encode_text_labels(model, all_types, device)
        
        # Encode image and text
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # Calculate similarity (softmax for probabilities)
            logits_per_image = model.logit_scale.exp() * image_features @ text_features.T
            probs = logits_per_image.softmax(dim=-1)
        
        # Calculate violation scores
        violation_scores = probs[0][:len(violation_types)]  # tensor of size K
        # Use MAX violation (robust) instead of SUM to avoid false positives
        max_violation_score, max_idx = (violation_scores.max().item(), int(violation_scores.argmax().item())) if len(violation_types) > 0 else (0.0, -1)
        top_violation_label = violation_types[max_idx] if 0 <= max_idx < len(violation_types) else None
        violation_sum = float(violation_scores.sum().item())
        
        # Healthy score
        healthy_score = probs[0][-1].item()
        
        # Thresholds and margin
        violation_threshold = float(os.environ.get("CLIP_VIOLATION_THRESHOLD", str(violation_threshold)))
        skip_threshold = float(os.environ.get("CLIP_SKIP_THRESHOLD", str(skip_threshold)))
        margin = float(os.environ.get("CLIP_MARGIN", "0.2"))

        # Determine if violation: require max violation above threshold AND above healthy by a margin
        is_violation = (max_violation_score >= violation_threshold) and ((max_violation_score - healthy_score) >= margin)

        # Determine if should skip processing (clear violation)
        skip_processing = (max_violation_score >= skip_threshold) and ((max_violation_score - healthy_score) >= margin)
        
        return {
            "is_violation": is_violation,
            "violation_score": max_violation_score,
            "healthy_score": healthy_score,
            "skip_processing": skip_processing,
            "method": "clip_prefilter",
            "violation_details": {
                type_name: float(score.item()) for type_name, score in zip(violation_types, violation_scores)
            },
            "violation_sum": violation_sum,
            "top_violation_label": top_violation_label
        }
    
    except Exception as e:
        # If CLIP fails, return safe defaults (don't skip processing)
        return {
            "is_violation": False,
            "violation_score": 0.0,
            "healthy_score": 1.0,
            "skip_processing": False,
            "method": "clip_prefilter",
            "error": str(e)
        }


def verify_caption_quality(
    image_path: str,
    caption: str,
    threshold: float = 0.7
) -> Dict[str, any]:
    """
    Verify caption có đúng với ảnh không bằng CLIP.
    
    Args:
        image_path: Đường dẫn đến ảnh
        caption: Caption cần verify
        threshold: Threshold để coi là caption đúng (default: 0.7)
    
    Returns:
        Dict với:
            - is_valid: bool - Caption có đúng không
            - similarity: float - Độ tương đồng (0-1)
            - method: str - "clip_verify"
    """
    if not os.path.exists(image_path) or not caption:
        return {
            "is_valid": False,
            "similarity": 0.0,
            "method": "clip_verify",
            "error": "Image not found or caption empty"
        }
    
    try:
        model, preprocess, device = _get_clip_model()
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Tokenize caption
        if clip is None:
            raise RuntimeError("CLIP module not available for tokenize")
        text_input = clip.tokenize([caption]).to(device)
        
        # Encode image and text
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_input)
            
            # Calculate similarity
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
        
        is_valid = similarity >= threshold
        
        return {
            "is_valid": is_valid,
            "similarity": similarity,
            "method": "clip_verify"
        }
    
    except Exception as e:
        # If CLIP fails, assume caption is valid (don't block processing)
        # Log error for debugging
        import sys
        print(f"[CLIP Verify Error] {e}", file=sys.stderr)
        return {
            "is_valid": True,
            "similarity": 0.0,
            "method": "clip_verify",
            "error": str(e)
        }


def classify_image_zeroshot(image_path: str, labels: list[str]) -> Dict[str, any]:
    """Zero-shot phân loại ảnh theo danh sách nhãn văn bản.
    Trả về nhãn cao nhất, điểm và phân phối xác suất.
    """
    if not os.path.exists(image_path) or not labels:
        return {"top_label": None, "top_score": 0.0, "scores": {}, "method": "clip_zeroshot_cls"}
    try:
        model, preprocess, device = _get_clip_model()
        # Encode image
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Encode labels (multilingual templates)
        text_features = _encode_text_labels(model, labels, device)
        with torch.no_grad():
            logits = model.logit_scale.exp() * (image_features @ text_features.T)  # [1, K]
            probs = logits.softmax(dim=-1)[0]  # [K]

        # Thresholding to avoid low-confidence/unstable labels (defaults tuned)
        min_score = float(os.environ.get("CLIP_LABEL_MIN_SCORE", "0.5"))
        min_gap = float(os.environ.get("CLIP_LABEL_MIN_GAP", "0.1"))
        margin_vs_healthy = float(os.environ.get("CLIP_LABEL_HEALTHY_MARGIN", "0.15"))
        strict = os.environ.get("CLIP_LABEL_STRICT", "1") == "1"

        top_idx = int(probs.argmax().item())
        top_label = labels[top_idx]
        top_score = float(probs[top_idx].item())

        # Compute second-best gap
        sorted_vals, sorted_idx = torch.sort(probs, descending=True)
        second_score = float(sorted_vals[1].item()) if sorted_vals.numel() > 1 else 0.0
        gap = top_score - second_score

        # Try to detect presence of a healthy label in provided labels
        healthy_candidates = [s for s in labels if s.lower().strip() in {
            "lành mạnh", "nội dung lành mạnh", "nội dung an toàn", "không vi phạm",
            "safe content", "no violation", "clean"
        }]
        healthy_score = 0.0
        if healthy_candidates:
            # take max over all healthy variants
            healthy_idx = [labels.index(s) for s in healthy_candidates if s in labels]
            if healthy_idx:
                healthy_score = max(float(probs[i].item()) for i in healthy_idx)

        # Suppress low-confidence predictions
        if strict and ((top_score < min_score) or (gap < min_gap) or (healthy_candidates and (top_score - healthy_score) < margin_vs_healthy)):
            return {"top_label": None, "top_score": top_score, "scores": {lbl: float(probs[i].item()) for i, lbl in enumerate(labels)}, "method": "clip_zeroshot_cls"}

        scores = {lbl: float(probs[i].item()) for i, lbl in enumerate(labels)}
        return {"top_label": top_label, "top_score": top_score, "scores": scores, "method": "clip_zeroshot_cls"}
    except Exception as e:
        return {"top_label": None, "top_score": 0.0, "scores": {}, "method": "clip_zeroshot_cls", "error": str(e)}

def detect_violation_type(
    image_path: str,
    violation_description: str
) -> Dict[str, any]:
    """
    Phát hiện loại vi phạm cụ thể bằng zero-shot CLIP.
    
    Args:
        image_path: Đường dẫn đến ảnh
        violation_description: Mô tả loại vi phạm (ví dụ: "quảng cáo thuốc lá")
    
    Returns:
        Dict với:
            - score: float - Điểm vi phạm (0-1)
            - is_violation: bool - Có phải vi phạm không
            - method: str - "clip_zeroshot"
    """
    if not os.path.exists(image_path):
        return {
            "score": 0.0,
            "is_violation": False,
            "method": "clip_zeroshot",
            "error": "Image not found"
        }
    
    try:
        model, preprocess, device = _get_clip_model()
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Tokenize violation description
        text_input = clip.tokenize([violation_description]).to(device)
        
        # Encode image and text
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_input)
            
            # Calculate similarity
            logits_per_image = model.logit_scale.exp() * image_features @ text_features.T
            score = logits_per_image.softmax(dim=-1)[0][0].item()
        
        # Threshold for violation (can be adjusted)
        violation_threshold = float(os.environ.get("CLIP_ZEROSHOT_THRESHOLD", "0.8"))
        is_violation = score >= violation_threshold
        
        return {
            "score": score,
            "is_violation": is_violation,
            "method": "clip_zeroshot"
        }
    
    except Exception as e:
        return {
            "score": 0.0,
            "is_violation": False,
            "method": "clip_zeroshot",
            "error": str(e)
        }

