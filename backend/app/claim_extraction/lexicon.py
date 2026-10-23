"""Lexical resources used by the rule-based claim extractor.

These lists are deliberately explicit so the extractor's behaviour is inspectable and testable.
"""

from __future__ import annotations

# The 80 COCO object categories (names as used by Ultralytics YOLO models trained on COCO).
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]  # fmt: skip

# Surface lemma / phrase -> canonical COCO class. Multi-word keys are matched on compound phrases.
SYNONYMS: dict[str, str] = {
    # people
    "person": "person", "people": "person", "man": "person", "woman": "person", "boy": "person",
    "girl": "person", "child": "person", "kid": "person", "baby": "person", "guy": "person", "lady": "person",
    "gentleman": "person", "player": "person", "skier": "person", "surfer": "person", "rider": "person",
    "pedestrian": "person", "toddler": "person", "teenager": "person", "adult": "person", "chef": "person",
    "worker": "person", "athlete": "person", "skateboarder": "person", "snowboarder": "person",
    "biker": "person", "cyclist": "person", "crowd": "person", "individual": "person", "human": "person",
    "men": "person", "women": "person", "children": "person", "policeman": "person", "officer": "person",
    "student": "person", "spectator": "person", "tourist": "person", "doctor": "person", "girlfriend": "person",
    # animals
    "dog": "dog", "puppy": "dog", "pup": "dog", "cat": "cat", "kitten": "cat", "kitty": "cat",
    "horse": "horse", "pony": "horse", "sheep": "sheep", "lamb": "sheep", "cow": "cow", "calf": "cow",
    "cattle": "cow", "bull": "cow", "elephant": "elephant", "bear": "bear", "zebra": "zebra",
    "giraffe": "giraffe", "bird": "bird", "pigeon": "bird", "seagull": "bird", "gull": "bird", "duck": "bird",
    "parrot": "bird", "goose": "bird", "chicken": "bird", "swan": "bird",
    # vehicles
    "bicycle": "bicycle", "bike": "bicycle", "car": "car", "automobile": "car", "taxi": "car",
    "suv": "car", "motorcycle": "motorcycle", "motorbike": "motorcycle", "scooter": "motorcycle",
    "airplane": "airplane", "plane": "airplane", "aeroplane": "airplane", "jet": "airplane",
    "aircraft": "airplane", "airliner": "airplane", "bus": "bus", "train": "train", "locomotive": "train",
    "truck": "truck", "lorry": "truck", "pickup": "truck", "boat": "boat", "ship": "boat", "sailboat": "boat",
    "canoe": "boat", "kayak": "boat", "yacht": "boat",
    # outdoor
    "traffic light": "traffic light", "stoplight": "traffic light", "fire hydrant": "fire hydrant",
    "hydrant": "fire hydrant", "stop sign": "stop sign", "parking meter": "parking meter", "bench": "bench",
    # accessories
    "backpack": "backpack", "umbrella": "umbrella", "handbag": "handbag", "purse": "handbag",
    "tie": "tie", "necktie": "tie", "suitcase": "suitcase", "luggage": "suitcase",
    # sports
    "frisbee": "frisbee", "ski": "skis", "skis": "skis", "snowboard": "snowboard", "ball": "sports ball",
    "football": "sports ball", "soccer ball": "sports ball", "basketball": "sports ball",
    "tennis ball": "sports ball", "baseball": "sports ball", "volleyball": "sports ball",
    "kite": "kite", "baseball bat": "baseball bat", "bat": "baseball bat", "baseball glove": "baseball glove",
    "glove": "baseball glove", "mitt": "baseball glove", "skateboard": "skateboard",
    "surfboard": "surfboard", "tennis racket": "tennis racket", "racket": "tennis racket",
    "racquet": "tennis racket",
    # kitchen
    "bottle": "bottle", "wine glass": "wine glass", "cup": "cup", "mug": "cup", "fork": "fork",
    "knife": "knife", "spoon": "spoon", "bowl": "bowl",
    # food
    "banana": "banana", "apple": "apple", "sandwich": "sandwich", "burger": "sandwich",
    "orange": "orange", "broccoli": "broccoli", "carrot": "carrot", "hot dog": "hot dog",
    "hotdog": "hot dog", "pizza": "pizza", "donut": "donut", "doughnut": "donut", "cake": "cake",
    "cupcake": "cake",
    # furniture
    "chair": "chair", "couch": "couch", "sofa": "couch", "potted plant": "potted plant",
    "houseplant": "potted plant", "plant": "potted plant", "bed": "bed", "table": "dining table",
    "dining table": "dining table", "desk": "dining table", "toilet": "toilet",
    # electronics
    "tv": "tv", "television": "tv", "monitor": "tv", "laptop": "laptop", "computer": "laptop",
    "notebook computer": "laptop", "mouse": "mouse", "remote": "remote", "remote control": "remote",
    "controller": "remote", "keyboard": "keyboard", "phone": "cell phone", "cellphone": "cell phone",
    "cell phone": "cell phone", "smartphone": "cell phone", "mobile phone": "cell phone",
    # appliances
    "microwave": "microwave", "oven": "oven", "stove": "oven", "toaster": "toaster", "sink": "sink",
    "refrigerator": "refrigerator", "fridge": "refrigerator",
    # indoor
    "book": "book", "clock": "clock", "vase": "vase", "scissors": "scissors", "teddy bear": "teddy bear",
    "teddy": "teddy bear", "hair drier": "hair drier", "hair dryer": "hair drier", "hairdryer": "hair drier",
    "toothbrush": "toothbrush",
}  # fmt: skip

CLASS_CATEGORY: dict[str, str] = {
    "person": "person",
    **{c: "animal" for c in ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]},
    **{c: "vehicle" for c in ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]},
    **{c: "food" for c in ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                           "donut", "cake"]},
}  # fmt: skip

# Generic category nouns: "an animal" refers to a category, not a specific class.
CATEGORY_NOUNS: dict[str, str] = {"animal": "animal", "vehicle": "vehicle", "food": "food", "pet": "animal"}

# Fine-grained person nouns that imply an identity claim beyond "a person is visible".
PERSON_SUBTYPES: set[str] = {
    "man", "woman", "boy", "girl", "child", "kid", "baby", "toddler", "teenager", "lady", "gentleman",
    "guy", "player", "skier", "surfer", "rider", "chef", "policeman", "officer", "athlete", "skateboarder",
    "snowboarder", "cyclist", "biker", "doctor", "student", "worker",
}  # fmt: skip

# Nouns denoting a setting / "stuff" rather than a countable object.
SCENE_NOUNS: set[str] = {
    "field", "park", "beach", "street", "road", "room", "kitchen", "bedroom", "bathroom", "living room",
    "forest", "woods", "city", "sky", "grass", "lawn", "snow", "water", "ocean", "sea", "lake", "river",
    "mountain", "mountains", "hill", "desert", "garden", "yard", "backyard", "sidewalk", "office",
    "restaurant", "airport", "market", "stadium", "court", "playground", "highway", "parking lot", "shore", "pool", "meadow", "farm", "pasture", "track", "runway", "station", "classroom", "store",
    "shop", "library", "hallway", "corridor", "living", "countryside", "village", "town", "downtown",
    "intersection", "trail", "path", "sand", "dirt", "floor", "ground", "wall", "sunset", "sunrise",
    "night", "daytime", "cloud", "clouds", "waves", "wave", "rain", "fog", "building", "buildings",
    "house", "skyline", "background", "outdoors", "indoors", "zoo", "enclosure", "slope", "rink", "arena",
}  # fmt: skip

# Nouns referring to the image itself or abstractions — never treated as visual entities.
NON_ENTITY_NOUNS: set[str] = {
    "image", "picture", "photo", "photograph", "scene", "view", "shot", "frame", "video", "clip", "moment",
    "time", "atmosphere", "mood", "life", "idea", "way", "fun", "day", "lot", "kind", "sort", "type",
    "variety", "number", "part", "side", "front", "top", "left", "right", "middle", "center", "back",
    "bottom", "edge", "corner", "distance", "foreground", "camera", "viewer", "something", "thing",
    "things", "area", "place", "spot", "color", "colour", "shade", "feeling", "beauty", "group", "pair",
    "couple", "bunch", "herd", "flock", "crowd", "set", "row", "pile", "stack", "piece", "slice",
}  # fmt: skip

# Collective nouns that introduce a plural entity: "a group of people" -> people (count >= 2).
COLLECTIVE_NOUNS: dict[str, tuple[str, int]] = {
    "group": ("at_least", 2), "bunch": ("at_least", 2), "herd": ("at_least", 2), "flock": ("at_least", 2),
    "crowd": ("at_least", 2), "pair": ("exact", 2), "couple": ("exact", 2), "row": ("at_least", 2),
    "pile": ("at_least", 2), "stack": ("at_least", 2), "set": ("at_least", 2), "team": ("at_least", 2),
}  # fmt: skip

CLOTHING_NOUNS: set[str] = {
    "shirt", "t-shirt", "tshirt", "jacket", "coat", "dress", "hat", "cap", "helmet", "jeans", "pants",
    "trousers", "shorts", "skirt", "sweater", "hoodie", "uniform", "suit", "glasses", "sunglasses",
    "shoes", "shoe", "boots", "boot", "scarf", "vest", "gloves", "jersey", "top", "blouse", "apron",
    "wetsuit", "costume", "outfit", "sneakers", "sandals", "beanie", "mask", "goggles",
}  # fmt: skip

# Color words -> set of acceptable color bins used by the pixel-level color analyzer.
COLOR_WORDS: dict[str, set[str]] = {
    "red": {"red"}, "crimson": {"red"}, "scarlet": {"red"}, "maroon": {"red", "brown"},
    "orange": {"orange"}, "yellow": {"yellow"}, "gold": {"yellow", "orange"}, "golden": {"yellow", "orange"},
    "green": {"green"}, "olive": {"green", "yellow"}, "lime": {"green"}, "teal": {"green", "blue"},
    "blue": {"blue"}, "navy": {"blue"}, "turquoise": {"blue", "green"}, "cyan": {"blue"},
    "purple": {"purple"}, "violet": {"purple"}, "lavender": {"purple"}, "magenta": {"pink", "purple"},
    "pink": {"pink"}, "brown": {"brown", "orange"}, "tan": {"brown", "orange"}, "beige": {"white", "brown"},
    "black": {"black"}, "white": {"white"}, "gray": {"gray"}, "grey": {"gray"}, "silver": {"gray", "white"},
}  # fmt: skip

# Adjectives expressing judgement, emotion or other non-visual properties.
SUBJECTIVE_ADJECTIVES: set[str] = {
    "beautiful", "pretty", "ugly", "cute", "lovely", "happy", "sad", "angry", "peaceful", "serene",
    "delicious", "tasty", "nice", "gorgeous", "amazing", "stunning", "boring", "interesting", "scary",
    "calm", "relaxed", "elegant", "majestic", "adorable", "friendly", "excited", "joyful", "cozy",
    "romantic", "important", "famous", "expensive", "cheap", "fun", "funny", "perfect", "wonderful",
    "great", "good", "bad", "lonely", "proud", "curious", "hungry", "tired", "bored", "busy", "lazy",
    "brave", "playful", "charming", "picturesque", "idyllic", "breathtaking", "fresh", "healthy", "smart",
    "loyal", "nervous", "content", "cheerful", "gentle", "fancy", "awesome", "magnificent", "vibrant",
    "delightful", "pleasant", "tranquil", "warm", "welcoming",
}  # fmt: skip

QUANTITY_ADJECTIVES: dict[str, tuple[str, int]] = {
    "several": ("at_least", 2), "many": ("at_least", 2), "multiple": ("at_least", 2),
    "numerous": ("at_least", 2), "various": ("at_least", 2), "some": ("at_least", 1),
    "few": ("at_least", 2), "both": ("exact", 2), "lots": ("at_least", 2),
}  # fmt: skip

NUMBER_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "a single": 1, "single": 1, "a dozen": 12, "dozen": 12,
}  # fmt: skip

MENTAL_VERBS: set[str] = {
    "think", "want", "love", "enjoy", "feel", "hope", "believe", "plan", "wish", "like", "prefer",
    "remember", "know", "decide", "try", "intend", "dream", "worry", "wonder", "miss", "hate", "need",
    "expect", "imagine", "pretend", "celebrate",
}  # fmt: skip

HEDGE_WORDS: set[str] = {
    "maybe", "perhaps", "possibly", "probably", "likely", "seemingly", "apparently", "presumably",
    "might", "may", "could", "seem", "appear", "suppose", "guess",
}  # fmt: skip

# Transitive verbs whose object participates in the action.
INTERACTION_VERBS: set[str] = {
    "hold", "carry", "ride", "wear", "eat", "drink", "throw", "catch", "kick", "hit", "push", "pull", "use",
    "read", "touch", "hug", "feed", "walk", "drive", "fly", "chase", "pet", "grab", "lift", "cut", "swing",
    "play", "watch", "look", "brush", "wash", "pour", "sit", "lie", "stand", "lean", "lay", "hold up",
    "type", "talk", "bite", "lick", "sniff", "fix", "paint", "climb", "cross", "pick", "open", "close",
    "kiss", "point", "serve", "slice", "cook", "hang", "sleep",
}  # fmt: skip

# Event verbs that describe a change over time and require multi-frame evidence.
TEMPORAL_VERBS: set[str] = {
    "enter", "leave", "exit", "appear", "disappear", "arrive", "depart", "approach", "start", "stop",
    "begin", "fall", "drop", "turn", "pick up", "put down", "open", "close", "emerge", "vanish",
    "join", "return", "come", "go", "move", "walk away", "run away", "get up", "sit down", "stand up",
}  # fmt: skip

# Verbs that only make sense as temporal events (not interpretable as a static depiction).
STRICT_TEMPORAL_VERBS: set[str] = {
    "enter", "leave", "exit", "appear", "disappear", "arrive", "depart", "emerge", "vanish", "return",
}  # fmt: skip

TEMPORAL_MARKERS: set[str] = {"then", "after", "before", "later", "finally", "first", "next", "afterwards",
                              "eventually", "suddenly", "meanwhile", "until", "while"}  # fmt: skip

MOTION_VERBS: set[str] = {"run", "walk", "move", "drive", "fly", "jump", "swim", "ride", "chase", "race",
                          "skate", "ski", "surf", "dance", "roll", "approach", "cross"}  # fmt: skip

# Prepositions mapped to normalized spatial relations.
SPATIAL_PREPOSITIONS: dict[str, str] = {
    "on": "on", "upon": "on", "on top of": "on", "atop": "on", "under": "under", "underneath": "under",
    "beneath": "under", "below": "below", "above": "above", "over": "above", "next to": "next_to",
    "beside": "next_to", "besides": "next_to", "by": "next_to", "near": "near", "close to": "near",
    "nearby": "near", "behind": "behind", "in front of": "in_front_of", "to the left of": "left_of",
    "left of": "left_of", "on the left of": "left_of", "to the right of": "right_of",
    "right of": "right_of", "on the right of": "right_of", "inside": "inside", "in": "inside",
    "within": "inside", "into": "inside", "between": "between", "against": "next_to", "around": "around",
    "across": "across", "along": "along", "through": "through", "at": "at", "outside": "outside",
}  # fmt: skip

# Relations that cannot be established from 2-D boxes alone (need depth / 3-D reasoning).
DEPTH_RELATIONS: set[str] = {"behind", "in_front_of"}

# Phrasing patterns that look like instructions to the verifier (prompt injection) rather than description.
INJECTION_PATTERNS: list[str] = [
    r"\bignore\b.{0,40}\b(instruction|prompt|rule|previous|above|prior)",
    r"\bdisregard\b.{0,40}\b(instruction|prompt|rule|previous|above|prior|evidence)",
    r"\b(system|developer)\s*(prompt|message|instruction)",
    r"\byou are (now|an?|the)\b.{0,40}\b(assistant|model|ai|verifier|system)",
    r"\b(mark|label|classify|rate|report|output|return|answer|respond)\b.{0,60}"
    r"\b(supported|contradicted|true|verified|correct|yes)\b",
    r"\b(SUPPORTED|CONTRADICTED|INSUFFICIENT_EVIDENCE|REVIEW_REQUIRED)\b",
    r"<\s*/?\s*(system|instruction|prompt|script)\b",
    r"\bnew instructions?\b",
    r"\boverride\b.{0,40}\b(verdict|result|policy|config|setting)",
    r"\bjailbreak\b",
]


def canonical_class(lemma: str) -> str | None:
    lemma = lemma.lower().strip()
    if lemma in SYNONYMS:
        return SYNONYMS[lemma]
    if lemma in COCO_CLASSES:
        return lemma
    return None


def category_of(canonical: str | None) -> str | None:
    if canonical is None:
        return None
    return CLASS_CATEGORY.get(canonical)
