"""
Generate conversation bridges using Gemini API.

Picks two consecutive messages from the real chat and asks Gemini to
generate a natural interchange between them — hallucinating realistic
daily-life experiences while perfectly matching the existing texting
style.  Personal facts (names, birthdays, etc.) are woven in so the
model learns them.

The bridge starts at message A, riffs on some everyday topic, and
naturally arrives at message B.  The result is spliced into the chat
data for training.

Usage:
    python gemini_generate.py \
        --input chat_history_train.txt \
        --output augmented_chats.txt \
        --api_key YOUR_GEMINI_API_KEY \
        --num_bridges 50
"""

import argparse
import random
import time
import requests
import json


# ─── PERSONAL FACTS ───────────────────────────────────────────────
# Fill these in. Some bridges will naturally reference them so the
# model learns who it's talking to/about.
PERSONAL_FACTS = {
    "M_name": "Matthew",          # full first name for M
    "I_name": "Iris",             # full first name for I
    "M_birthday": "November 21",      # e.g. "March 15"
    "I_birthday": "March 24",      # e.g. "July 22"
    "M_school": "UT Austin",        # e.g. "UNT"
    "I_school": "Stanford",        # e.g. "UNT"
    "M_major": "Math",         # e.g. "CS"
    "I_major": "Mechanical Engineering",         # e.g. "Bio"
    "I_hobby": "Catan, Salem, Stardew Valley, Gardening",
    "M_hobby": "Starcraft II, Coding, Physics, History",
}

# ─── LIFE FACTS ──────────────────────────────────────────────────
# ALL of these get pasted into every prompt as background context.
# Gemini will naturally weave in whichever ones are relevant to the
# randomly-selected daily topic.  Add as many as you want.
LIFE_FACTS = [
    # --- Food / Restaurants ---
    "Matthew likes bananas and chocolate",
    "Iris likes fried chicken and chocolate",

    # --- School / Classes ---
    "Matthew studies math",
    "Iris studies mechanical engineering",

    # --- Music / Hobbies ---
    "Matthew's hobbies are Starcraft II, Coding, Physics, History",
    "Iris'hobbies are Catan, Salem, Stardew Valley, Gardening",
    "Matthew loves talking to iris about things she likes",

    # --- Places ---
    "Matthew lives at Stanford University and spends a lot of late nights in the math lounge",
    "Iris lives at UT Austin",

    # --- Pets / Animals ---
    "Iris wants a cat but her apartment doesn't allow it",

    # --- Random facts ---
    "Matthew loves bringing up random idioms when it only loosely makes sense.",
    "Matthew has a dog named Charlie and a cat named Ellie whom Iris adores.",
    "Iris watches cooking videos a lot on youtube",
    "Matthew listens to the Police, the Beatles, and David Bowie",
    "Iris listens to lana del ray, taylor swift, and pop music"
]

# ─── DAILY TOPICS ────────────────────────────────────────────────
# One is randomly picked per bridge to drive the conversation.
# Gemini combines it with whatever life facts fit naturally.
DAILY_TOPICS = [
    "The names of the two conversers are Matthew Li and Iris Nguyen (whats your name?)",
    "The names of the two conversers are Matthew Li and Iris Nguyen (whats your name?)",
    "The birthdays of the two conversers (whens your bday/birthday?, What are you getting?)",
    "what they ate today",
    "plans for the day",
    "something funny or weird that happened today",
    "something funny or weird that happened today",
    "something funny or weird that happened today",
    "a show, movie, or youtube video one of them watched",
    "music they're listening to or a song stuck in their head",
    "working out, gym, running, or being lazy about it",
    "Matthew staying up way too late being excited about something random like coding a project",
    "shopping, wanting to buy something, or being broke",
    "social media — something they saw on instagram, etc.",
    "a pet, animal, or wanting a pet",
    "an embarrassing or clumsy moment",
    "someone they both know did something funny or annoying",
    "travel plans or somewhere they want to go (National Parks)",
]

def load_chat_file(path):
    """Load the raw chat .txt file and return lines."""
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def find_insertion_points(lines, min_msg_len=5):
    """
    Find good points where we can insert a knowledge bridge.

    We look for speaker transitions (I→M or M→I) where both
    messages have enough substance to anchor a bridge.
    Returns list of line indices — a bridge would be inserted
    BETWEEN lines[idx-1] and lines[idx].
    """
    points = []
    for i in range(1, len(lines)):
        prev = lines[i - 1].strip()
        curr = lines[i].strip()

        # Must be a speaker transition
        prev_is_i = prev.startswith("I: ")
        prev_is_m = prev.startswith("M: ")
        curr_is_i = curr.startswith("I: ")
        curr_is_m = curr.startswith("M: ")

        if not ((prev_is_i and curr_is_m) or (prev_is_m and curr_is_i)):
            continue

        # Both messages should have some substance (not just emoji)
        prev_text = prev[3:]
        curr_text = curr[3:]
        if len(prev_text) < min_msg_len or len(curr_text) < min_msg_len:
            continue

        points.append(i)

    return points


def select_spaced_points(points, num_needed, min_spacing=50):
    """
    Select insertion points with minimum spacing between them
    so bridges don't cluster together.
    """
    shuffled = list(points)
    random.shuffle(shuffled)

    selected = []
    for pt in shuffled:
        if all(abs(pt - s) >= min_spacing for s in selected):
            selected.append(pt)
        if len(selected) >= num_needed:
            break

    selected.sort()
    return selected


def extract_style_examples(lines, num_examples=6, chunk_size=40):
    """
    Extract random conversation chunks to demonstrate the texting style.
    Returns longer chunks so Gemini really absorbs the voice.
    """
    examples = []
    m_starts = [i for i, line in enumerate(lines) if line.startswith("M: ")]

    if not m_starts:
        return []

    sampled = random.sample(m_starts, min(num_examples * 4, len(m_starts)))

    for start in sampled:
        end = min(start + chunk_size, len(lines))
        chunk = lines[start:end]
        has_i = any(l.startswith("I: ") for l in chunk)
        has_m = any(l.startswith("M: ") for l in chunk)
        if has_i and has_m and len(chunk) >= 15:
            examples.append("".join(chunk).strip())
            if len(examples) >= num_examples:
                break

    return examples


def _facts_block():
    """Build the full facts context block from PERSONAL_FACTS + LIFE_FACTS."""
    lines = []
    f = PERSONAL_FACTS
    if f.get("M_name") and f.get("I_name"):
        lines.append(f"- M's real name is {f['M_name']}. I's real name is {f['I_name']}.")
    if f.get("M_birthday", "FILL_IN") != "FILL_IN":
        lines.append(f"- {f['M_name']}'s birthday is {f['M_birthday']}.")
    if f.get("I_birthday", "FILL_IN") != "FILL_IN":
        lines.append(f"- {f['I_name']}'s birthday is {f['I_birthday']}.")
    if f.get("M_school", "FILL_IN") != "FILL_IN":
        lines.append(f"- They go to {f['M_school']}.")
    if f.get("M_major", "FILL_IN") != "FILL_IN" and f.get("I_major", "FILL_IN") != "FILL_IN":
        lines.append(f"- {f['M_name']} studies {f['M_major']}. {f['I_name']} studies {f['I_major']}.")
    # Extra personal facts key-value pairs
    skip = {"M_name", "I_name", "M_birthday", "I_birthday", "M_school", "I_school", "M_major", "I_major"}
    for k, v in PERSONAL_FACTS.items():
        if k not in skip and v != "FILL_IN":
            lines.append(f"- {k}: {v}")
    # Life facts (all of them, so Gemini can draw from any)
    for lf in LIFE_FACTS:
        if lf.strip():
            lines.append(f"- {lf}")
    return "\n".join(lines) if lines else ""


def build_bridge_prompt(style_examples, msg_before, msg_after,
                        context_before, context_after, topic=None):
    """
    Build the Gemini prompt for generating a daily-life bridge.
    """
    # Style examples block
    style_text = ""
    for i, ex in enumerate(style_examples, 1):
        style_text += f"\n--- Style Example {i} ---\n{ex}\n"

    context_before_text = "".join(context_before).strip()
    context_after_text = "".join(context_after).strip()

    facts = _facts_block()
    facts_section = f"\nFACTS ABOUT THESE TWO PEOPLE (weave in naturally when relevant):\n{facts}\n" if facts else ""

    prompt = f"""You are generating training data for a chatbot that texts like a real person. Two friends — I and M — text each other constantly. They're witty, playful, and curious. Have a good bit of question answer schemes.

Overall, your goal is to do STYLE TRANSFER so MAINTAIN THE STYLE IN THE EXAMPLES. This includes:
- Every line is either "I: <message>" or "M: <message>"
- Very casual. ENSURE IDENTICAL STYLE TO EXAMPLES
- SIMILAR emoji use: 😭😂🙈💀‼️⁉️🤯 etc.
- Similar typos and formatting
- Same distribution of I: and M: as examples
- LOTS of playful roasting, banter, one-upping each other, especially from M.
- Random idioms from M when they only loosely fit. Examples: One bird in the hand is better than two in the bush.
- Simple vocabulary, teenage/young-adult energy
{facts_section}
Here are {len(style_examples)} REAL examples of their texting style — study these carefully:
{style_text}
--- End of style examples ---

CONTEXT — here is what was being discussed right before and after the insertion point:

[Before]:
{context_before_text}

[After]:
{context_after_text}

YOUR TASK:
Generate a conversation that starts with this EXACT message:
{msg_before.strip()}

And ends with this EXACT message:
{msg_after.strip()}

{f"In between, hallucinate a realistic daily-life experience related to: **{topic}**." if topic else "In between, continue the conversation naturally — let it spin off from the context above however feels most realistic."}
Imagine what would ACTUALLY happen between these two people — maybe one of them tells a story about something that happened, they react to it, joke about it, give each other a hard time, and naturally move on. Where it fits, reference the facts listed above so the model learns them. It should read exactly like a real text thread between two friends just living their lives.

CRITICAL RULES:
1. First line must be EXACTLY: {msg_before.strip()}
2. Last line must be EXACTLY: {msg_after.strip()}
3. Generate roughly 15 messages in between (the bridge)
4. ONLY output "I: " and "M: " lines — NO headers, labels, timestamps, or explanations
5. Do NOT wrap output in code blocks or markdown
6. The conversation must flow NATURALLY from start to end — no forced transitions
7. Keep the SAME energy, humor, and style as the examples above

Generate the bridging conversation now:"""

    return prompt


def call_gemini(prompt, api_key, model="gemini-2.0-flash",
                temperature=1.0, max_tokens=4096):
    """Call the Gemini API and return the generated text."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
            "topK": 40,
        },
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip()
    except (KeyError, IndexError) as e:
        print(f"  Error parsing response: {e}")
        print(f"  Response: {json.dumps(data, indent=2)[:500]}")
        return None


def clean_generated_text(text):
    """Keep only lines that start with 'I: ' or 'M: '."""
    cleaned = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("I: ") or line.startswith("M: "):
            cleaned.append(line)
    return cleaned


def validate_bridge(bridge_lines, msg_before, msg_after):
    """
    Make sure the bridge starts and ends with the correct anchor messages.
    Fix it if Gemini didn't follow instructions exactly.
    Returns None if the bridge is too short to be useful.
    """
    if not bridge_lines:
        return None

    # Ensure correct anchors
    if bridge_lines[0].strip() != msg_before.strip():
        bridge_lines.insert(0, msg_before.strip())
    if bridge_lines[-1].strip() != msg_after.strip():
        bridge_lines.append(msg_after.strip())

    # Need at least a handful of messages to be a useful bridge
    if len(bridge_lines) < 6:
        return None

    return bridge_lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate conversation bridges with world knowledge via Gemini"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to real chat .txt file")
    parser.add_argument("--output", type=str, default="augmented_chats.txt",
                        help="Output path for augmented chat file")
    parser.add_argument("--api_key", type=str, required=True,
                        help="Gemini API key")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Gemini model name")
    parser.add_argument("--num_bridges", type=int, default=20,
                        help="Number of knowledge bridges to generate")
    parser.add_argument("--style_examples", type=int, default=6,
                        help="Number of style examples per prompt")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--context_window", type=int, default=10,
                        help="Lines of context before/after insertion point")
    parser.add_argument("--min_spacing", type=int, default=50,
                        help="Minimum line spacing between insertion points")
    args = parser.parse_args()

    # --- Count life facts ---
    life_facts = [f for f in LIFE_FACTS if f.strip()]
    print(f"Loaded {len(life_facts)} life facts (injected into every prompt)")

    # --- Load data ---
    print(f"Loading chat data from {args.input}...")
    lines = load_chat_file(args.input)
    print(f"  {len(lines)} lines loaded")

    # --- Find & select insertion points ---
    print("Finding insertion points...")
    all_points = find_insertion_points(lines)
    print(f"  {len(all_points)} candidate points found")

    num_bridges = min(args.num_bridges, len(all_points))
    selected = select_spaced_points(all_points, num_bridges, args.min_spacing)
    print(f"  {len(selected)} points selected (min spacing {args.min_spacing})")

    if not selected:
        print("No valid insertion points — check your input file.")
        return

    # --- Generate bridges ---
    print(f"\nGenerating {len(selected)} bridges...")
    print(f"  Model: {args.model}  Temp: {args.temperature}  "
          f"Style examples: {args.style_examples}\n")

    bridges = []  # (insert_at, bridge_lines)

    for idx, insert_at in enumerate(selected):
        msg_before = lines[insert_at - 1]
        msg_after = lines[insert_at]

        # Surrounding context
        ctx_start = max(0, insert_at - args.context_window)
        ctx_end = min(len(lines), insert_at + args.context_window)
        context_before = lines[ctx_start:insert_at]
        context_after = lines[insert_at:ctx_end]

        # Style examples from elsewhere in the file
        style_examples = extract_style_examples(
            lines, num_examples=args.style_examples
        )

        # 50/50: either pick a daily topic or let Gemini spin off the context
        if random.random() < 0.5:
            topic = random.choice(DAILY_TOPICS)
        else:
            topic = None

        prompt = build_bridge_prompt(
            style_examples, msg_before, msg_after,
            context_before, context_after, topic,
        )

        try:
            raw = call_gemini(
                prompt, api_key=args.api_key,
                model=args.model, temperature=args.temperature,
            )

            if raw is None:
                print(f"  [{idx+1}/{len(selected)}] FAILED — no output")
                continue

            bridge = clean_generated_text(raw)
            bridge = validate_bridge(bridge, msg_before, msg_after)

            if bridge is None:
                print(f"  [{idx+1}/{len(selected)}] SKIPPED — too short")
                continue

            bridges.append((insert_at, bridge))
            label = topic[:60] if topic else "freeform"
            print(f"  [{idx+1}/{len(selected)}] ✓ {len(bridge)} msgs "
                  f"({label})")

        except requests.exceptions.HTTPError as e:
            print(f"  [{idx+1}/{len(selected)}] API ERROR: {e}")
            if "429" in str(e):
                print("    Rate limited — waiting 30s...")
                time.sleep(30)
            continue
        except Exception as e:
            print(f"  [{idx+1}/{len(selected)}] ERROR: {e}")
            continue

        time.sleep(args.delay)

    # --- Splice bridges into the original text ---
    print(f"\nSplicing {len(bridges)} bridges into chat...")
    augmented = list(lines)

    # Work backwards so earlier indices stay valid
    for insert_at, bridge_lines in reversed(bridges):
        # The bridge already contains the anchor messages (first & last line),
        # so we replace lines[insert_at-1 : insert_at+1] with the full bridge.
        bridge_text = [line + "\n" for line in bridge_lines]
        augmented[insert_at - 1 : insert_at + 1] = bridge_text

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(augmented)

    new_lines = sum(len(b) for _, b in bridges)
    replaced = 2 * len(bridges)  # two anchor lines per bridge
    print(f"\nDone!")
    print(f"  Bridges generated: {len(bridges)}/{len(selected)}")
    print(f"  Original lines:   {len(lines)}")
    print(f"  Augmented lines:  {len(augmented)}")
    print(f"  Net new lines:    {new_lines - replaced}")
    print(f"  Output:           {args.output}")


if __name__ == "__main__":
    main()
