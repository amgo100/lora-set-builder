"""Shot lists, prompts and captions.

A *take* is one planned image in the training set: framing, camera angle,
expression, outfit, setting and lighting. The default shot list spreads these
so that no two takes repeat the same combination and the set is balanced
between close-ups, medium shots and full-body shots.
"""

from __future__ import annotations

from typing import Any

from .project import new_id, normalise_take, slug

FRAMINGS = {
    "close-up": "close-up portrait, head and shoulders",
    "medium": "medium shot, waist up",
    "full-body": "full-body shot, head to feet visible",
}

ANGLES = [
    "facing the camera",
    "three-quarter view turned to the left",
    "three-quarter view turned to the right",
    "left profile",
    "right profile",
    "seen slightly from above",
    "seen slightly from below",
    "looking over the shoulder",
]

EXPRESSIONS = [
    "neutral expression",
    "gentle smile",
    "broad smile showing teeth",
    "serious, focused expression",
    "laughing",
    "thoughtful expression, looking away from the camera",
]

SETTINGS = [
    "plain light-grey studio backdrop",
    "city street with shops in the background",
    "park with trees and grass",
    "cosy cafe interior",
    "living room at home",
    "office with a desk and a window",
    "beach with the sea in the background",
    "forest trail",
    "kitchen",
    "library with bookshelves",
    "train station platform",
    "city street at night with neon signs",
]

LIGHTING = [
    "soft daylight",
    "golden hour sunlight",
    "overcast diffuse light",
    "studio softbox lighting",
    "warm indoor lamp light",
    "night-time street lighting",
]

# Detailed on purpose: with a vague outfit ("a hoodie") the model tends to copy
# the clothes from the reference images. Colour, material, fit and footwear
# give it something concrete to draw instead. Kept gender-neutral and free of
# hats and glasses, which would hide or alter identity features. Everything
# worn above the waist comes before the first comma: close-up and medium-shot
# captions keep only that part (see caption_outfit). The length
# (16) must stay coprime with the stride used below (3).
OUTFITS = [
    "a fitted white crew-neck t-shirt, dark indigo straight-leg jeans and white canvas sneakers",
    "a light-blue oxford button-up shirt with the sleeves rolled up, beige chinos and brown "
    "leather loafers",
    "a chunky cream cable-knit sweater, charcoal wool trousers and brown suede ankle boots",
    "a faded mid-blue denim jacket over a grey marl t-shirt, black jeans and black leather boots",
    "a tailored navy two-piece suit with a crisp white shirt, polished black leather shoes",
    "a black zip-up track jacket with white stripes down the sleeves, matching track pants and "
    "grey running shoes",
    "a long camel wool overcoat worn open with a red-and-green tartan scarf, dark grey trousers "
    "and black leather boots",
    "an oversized forest-green hoodie, light-grey jogger trousers and white high-top sneakers",
    "a mustard-yellow corduroy overshirt over a black t-shirt, olive cargo trousers and tan "
    "work boots",
    "a short-sleeved pale terracotta linen shirt, stone-coloured linen shorts and brown "
    "leather sandals",
    "a black leather biker jacket over a plain white t-shirt, slim black jeans and black "
    "Chelsea boots",
    "a quilted olive-green puffer jacket zipped up over a navy fleece, grey jeans and brown "
    "hiking boots",
    "a burgundy roll-neck sweater under a grey houndstooth wool blazer, dark trousers and "
    "brown brogues",
    "a navy-and-white striped Breton top, a beige trench coat worn open, dark jeans and white "
    "leather sneakers",
    "a bright-red hooded rain jacket with the hood down, navy trousers and yellow rubber boots",
    "a dark-green waxed cotton field jacket over a cream knitted jumper, brown corduroy "
    "trousers and leather lace-up boots",
]

# share of each framing in the default set
FRAMING_MIX = [("close-up", 0.35), ("medium", 0.40), ("full-body", 0.25)]


def default_shot_list(n: int = 20, keep_outfit: bool = False) -> list[dict[str, Any]]:
    """Build n takes that vary every field.

    Indices step through each list with different strides (1, 3, 5, 7, ...),
    which spreads combinations instead of walking the lists in lock-step.
    """
    framings: list[str] = []
    for name, share in FRAMING_MIX:
        framings += [name] * max(1, round(n * share))
    framings = (framings * 2)[:n]
    # interleave so the list does not start with all close-ups
    framings = framings[0::3] + framings[1::3] + framings[2::3]

    takes = []
    for i in range(n):
        framing = framings[i]
        angle = ANGLES[i % len(ANGLES)]
        # profile and over-the-shoulder shots read badly as full-body shots
        if framing == "full-body" and ("profile" in angle or "shoulder" in angle):
            angle = ANGLES[(i + 1) % 3]
        take = {
            "id": new_id("t"),
            "framing": framing,
            "angle": angle,
            "expression": EXPRESSIONS[(i * 5) % len(EXPRESSIONS)],
            "outfit": "" if keep_outfit else OUTFITS[(i * 3) % len(OUTFITS)],
            "setting": SETTINGS[(i * 5) % len(SETTINGS)],
            "lighting": LIGHTING[(i * 7) % len(LIGHTING)],
            "aspect": "square" if framing == "close-up" else "portrait",
        }
        if framing == "medium" and i % 4 == 1:
            take["aspect"] = "landscape"
        take["title"] = f"{framing} / {take['angle'].split(',')[0]}"
        takes.append(normalise_take(take))
    return takes


def anchor_take() -> dict[str, Any]:
    """Text-to-image take that invents the character from the description."""
    return normalise_take(
        {
            "id": "anchor-" + new_id(),
            "title": "Anchor: create the character",
            "role": "anchor",
            "mode": "t2i",
            "framing": "medium",
            "angle": "facing the camera",
            "expression": "neutral expression",
            "setting": "plain light-grey studio backdrop",
            "lighting": "soft daylight",
            "aspect": "portrait",
        }
    )


def turnaround_take() -> dict[str, Any]:
    """Reference take: a character sheet that adds side and back views."""
    return normalise_take(
        {
            "id": "sheet-" + new_id(),
            "title": "Reference sheet: turnaround",
            "role": "ref",
            "mode": "reference",
            "aspect": "landscape",
            "prompt_override": (
                "Character reference sheet of {subject_ref}: the same person shown four times "
                "side by side, full body, standing, on a plain light-grey background: front view, "
                "three-quarter view, side profile and back view. Even studio lighting. "
                "{identity_lock}"
            ),
        }
    )


# --------------------------------------------------------------------- prompts

IDENTITY_LOCK = {
    "person": (
        "Keep the face, facial features, skin tone, hairline, hair and body proportions "
        "exactly the same as in the reference images. Realistic photograph, natural skin "
        "texture, sharp focus."
    ),
    "character": (
        "Keep the character's face, features, colouring, hair, body proportions and "
        "rendering style exactly the same as in the reference images."
    ),
}


def _ref_tags(n_refs: int, offset: int = 0) -> str:
    tags = [f"<image{i + 1 + offset}>" for i in range(n_refs)]
    if not tags:
        return ""
    if len(tags) == 1:
        return tags[0]
    return ", ".join(tags[:-1]) + " and " + tags[-1]


def _scene(take: dict[str, Any]) -> str:
    parts = []
    if take.get("framing"):
        parts.append(FRAMINGS.get(take["framing"], take["framing"]))
    for key in ("angle", "expression"):
        if take.get(key):
            parts.append(take[key])
    sentence = ", ".join(parts)
    out = [sentence[:1].upper() + sentence[1:] + "." if sentence else ""]
    if take.get("outfit"):
        out.append(f"Wearing {take['outfit']}.")
    if take.get("setting"):
        out.append(f"Setting: {take['setting']}.")
    if take.get("lighting"):
        out.append(f"Lighting: {take['lighting']}.")
    if take.get("extra"):
        out.append(take["extra"].strip())
    return " ".join(p for p in out if p)


def compose_prompt(take: dict[str, Any], settings: dict[str, Any], n_refs: int) -> str:
    """Prompt sent to Qwen-Image-2.1 for one take.

    reference mode: image1..imageN are identity references.
    edit mode: image1 is the photo to change, image2.. are identity references.
    t2i mode: no images; the description defines the subject.
    """
    kind = settings.get("subject_kind", "person")
    who = "person" if kind == "person" else "character"
    lock = IDENTITY_LOCK.get(kind, IDENTITY_LOCK["person"])
    desc = settings.get("description", "").strip()
    mode = take.get("mode", "reference")

    if mode == "t2i":
        subject_ref = desc or f"a {settings.get('class_word', 'person')}"
    elif mode == "edit":
        subject_ref = f"the {who} in <image1>"
    else:
        subject_ref = f"the {who} shown in {_ref_tags(n_refs)}"

    if take.get("prompt_override"):
        return take["prompt_override"].format(
            subject_ref=subject_ref, identity_lock=lock, description=desc
        )

    scene = _scene(take)
    if mode == "t2i":
        style = (
            "Realistic photograph." if kind == "person" else ""
        )
        return f"{subject_ref}. {scene} {style}".strip()
    if mode == "edit":
        extra_refs = ""
        if n_refs > 1:
            extra_refs = (
                f" Use {_ref_tags(n_refs - 1, offset=1)} as additional references "
                f"for the {who}'s identity."
            )
        return (
            f"Edit <image1>: keep the same {who} and change the pose, clothing and "
            f"surroundings as follows. {scene}{extra_refs} {lock}"
        )
    clothes = ""
    if take.get("outfit"):
        # without this the model often keeps the clothes from the references
        clothes = (
            f" Dress the {who} in the outfit described above, not in the clothes "
            f"from the reference images."
        )
    return f"A new photograph of {subject_ref}. {scene}{clothes} {lock}"


def compose_refine_prompt(correction: str, settings: dict[str, Any]) -> str:
    """Prompt for refining one generated variant: <image1> is that variant.

    No identity lock and no other references on purpose: the correction may
    change exactly what the lock would hold fixed (hair, glasses, face shape).
    """
    who = "person" if settings.get("subject_kind", "person") == "person" else "character"
    fix = correction.strip().rstrip(".")
    return (
        f"Edit <image1>: {fix}. Change only that and keep everything else in <image1> "
        f"exactly as it is: the same {who}, pose, framing, background, lighting and "
        f"image style."
    )


def compose_caption(take: dict[str, Any], settings: dict[str, Any]) -> str:
    """Training caption: trigger + class, then only what varies between images.

    Identity features (face, hair colour...) are left out on purpose so the
    LoRA ties them to the trigger word instead of to the caption text.
    """
    trigger = settings.get("trigger", "").strip()
    cls = settings.get("class_word", "person").strip()
    parts = [f"{trigger} {cls}".strip()]
    if take.get("framing"):
        parts.append(FRAMINGS.get(take["framing"], take["framing"]).split(",")[0])
    for key in ("angle", "expression"):
        if take.get(key):
            parts.append(take[key])
    outfit = caption_outfit(take)
    if outfit:
        parts.append(f"wearing {outfit}")
    if take.get("setting"):
        parts.append(take["setting"])
    if take.get("lighting"):
        parts.append(take["lighting"])
    return ", ".join(parts)


def caption_outfit(take: dict[str, Any]) -> str:
    """The part of the outfit a caption should mention.

    Close-ups and medium shots end at the shoulders or the waist, so only the
    first part of the outfit (up to the first comma: the top, as the default
    outfits are written) is visible; trousers and shoes would describe
    things that are not in the image.
    """
    outfit = (take.get("outfit") or "").strip()
    if take.get("framing") in ("close-up", "medium"):
        outfit = outfit.split(",")[0].strip()
    return outfit


def source_caption(settings: dict[str, Any]) -> str:
    trigger = settings.get("trigger", "").strip()
    cls = settings.get("class_word", "person").strip()
    return f"{trigger} {cls}, photograph".strip()


def output_size(aspect: str, long_side: int) -> tuple[int, int]:
    from .project import ASPECTS

    a, b = ASPECTS.get(aspect, (1, 1))
    long_side = int(long_side) // 32 * 32
    if a >= b:
        w, h = long_side, long_side * b / a
    else:
        w, h = long_side * a / b, long_side
    return int(w) // 32 * 32, int(h) // 32 * 32


__all__ = [
    "default_shot_list",
    "anchor_take",
    "turnaround_take",
    "compose_prompt",
    "compose_refine_prompt",
    "compose_caption",
    "source_caption",
    "output_size",
    "slug",
]
