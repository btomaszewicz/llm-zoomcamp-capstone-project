def build_prompt_with_mode(question, context, prompt_mode):
    extra = PROMPT_MODES.get(prompt_mode, "")
    system_instructions = BASE_INSTRUCTIONS + "\n\n" + extra

    prompt = f"""
{system_instructions}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    return prompt