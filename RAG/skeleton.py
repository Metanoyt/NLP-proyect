model = ""
system_prompt = """You are an senior pytorch developer, your task is to review the code provided and suggest improvements or optimizations where necessary. Provide clear explanations for your suggestions."""

def generate_prompt(code_snippet):
    prompt = f"""{system_prompt}
    Here is a code snippet for your review:
    {code_snippet}
    Please provide your feedback and suggestions."""
    return prompt

def review_code(code_snippet):
    prompt = generate_prompt(code_snippet)
    # Here you would integrate with the language model to get the review
    # For example:
    # response = language_model.generate(prompt)
    # return response
    return prompt  # Placeholder return for demonstration purposes


Token: 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