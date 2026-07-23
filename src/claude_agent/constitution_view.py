"""
Rule-level views of CONSTITUTION.md. Single source of truth stays in the
.md file -- each rule is tagged inline with the scopes it requires (e.g.
"[FOOTAGE]" for rules only usable by something that can inspect actual
frames/audio). This module reads those tags rather than maintaining a
separate list that could drift out of sync as rules are added or changed.

To scope a rule, add its tag right after the rule ID in CONSTITUTION.md:
    R1.1 [FOOTAGE] — No jump cuts...
An untagged rule is visible to every role by default.

Format expected (condensed style): section headers are unindented lines
"N. Title", rules are indented lines "    R#.# [TAG] — text", one rule per
line.
"""

import re

# Tag that marks a rule as requiring actual footage/audio inspection --
# Producer is text-only and must never see or cite these.
FOOTAGE_TAG = "FOOTAGE"

_SECTION_RE = re.compile(r"^(\d+)\.\s+(.+?)\s*$")
_RULE_RE = re.compile(r"^\s+(R\d+\.\d+)((?:\s+\[\w+\])*)\s+[—-]\s+(.+?)\s*$")

# Section 0 (Beat archetypes) is a plain bulleted vocabulary list, not
# "R#.#" rules, and isn't subject to tag-based exclusion -- it's always
# included verbatim, unconditionally, for every role.
ALWAYS_INCLUDE_SECTIONS = {0}


def parse_rules(constitution_path="CONSTITUTION.md"):
    """Returns a list of dicts, one per rule: {id, tags, section, section_title, text}."""
    with open(constitution_path) as f:
        lines = f.read().splitlines()

    rules = []
    current_section = None
    current_title = None
    for line in lines:
        if not line.startswith((" ", "\t")):
            m = _SECTION_RE.match(line)
            if m:
                current_section = int(m.group(1))
                current_title = m.group(2)
                continue
        m = _RULE_RE.match(line)
        if m and current_section is not None:
            rule_id, tag_block, body = m.group(1), m.group(2), m.group(3)
            tags = set(re.findall(r"\[(\w+)\]", tag_block))
            rules.append({
                "id": rule_id,
                "tags": tags,
                "section": current_section,
                "section_title": current_title,
                "text": f"{rule_id}{tag_block} — {body}",
            })
    return rules


def print_rules(constitution_path="CONSTITUTION.md"):
    """Print every rule with its ID and tags, one per block, for manual review."""
    current_section = None
    for r in parse_rules(constitution_path):
        if r["section"] != current_section:
            current_section = r["section"]
            print(f"\n=== Section {current_section}: {r['section_title']} ===\n")
        tag_str = f" {sorted(r['tags'])}" if r["tags"] else ""
        print(f"[{r['id']}]{tag_str}  {r['text']}")


def constitution_view(excluded_tags: set[str], constitution_path="CONSTITUTION.md") -> str:
    """Rebuild the constitution text, dropping any rule carrying one of the
    excluded tags, keeping section structure intact (a section with every
    rule excluded is dropped too)."""
    with open(constitution_path) as f:
        text = f.read()
    lines = text.splitlines()

    # preamble = everything before the first unindented "N. " section header
    # whose number isn't in ALWAYS_INCLUDE_SECTIONS -- so Section 0 (Beat
    # archetypes) rides along in the preamble, verbatim, always.
    preamble_end = 0
    for i, line in enumerate(lines):
        if not line.startswith((" ", "\t")):
            m = _SECTION_RE.match(line)
            if m and int(m.group(1)) not in ALWAYS_INCLUDE_SECTIONS:
                preamble_end = i
                break
    preamble = "\n".join(lines[:preamble_end]).rstrip() + "\n"

    rules = parse_rules(constitution_path)
    out = [preamble]
    current_section = None
    for r in rules:
        if r["tags"] & excluded_tags:
            continue
        if r["section"] != current_section:
            current_section = r["section"]
            out.append(f"\n{r['section']}. {r['section_title']}\n")
        out.append(f"    {r['text']}\n")
    return "".join(out)


def producer_constitution(constitution_path="CONSTITUTION.md") -> str:
    return constitution_view({FOOTAGE_TAG}, constitution_path)
