import re
from pathlib import Path
from string import Formatter

PROMPT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*$")
PLACEHOLDER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class PromptLoader:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()

    def render(self, prompt_id: str, variables: dict[str, str]) -> str:
        if not PROMPT_ID_PATTERN.fullmatch(prompt_id):
            raise ValueError(f"invalid prompt ID: {prompt_id}")

        prompt_path = (self.root / f"{prompt_id.replace('.', '/')}.md").resolve()
        if not prompt_path.is_relative_to(self.root):
            raise ValueError(f"prompt ID escapes prompt root: {prompt_id}")

        template = prompt_path.read_text(encoding="utf-8")
        required = self._placeholders(template)
        supplied = set(variables)
        if required != supplied:
            missing = sorted(required - supplied)
            extra = sorted(supplied - required)
            details = []
            if missing:
                details.append(f"missing variables: {', '.join(missing)}")
            if extra:
                details.append(f"unexpected variables: {', '.join(extra)}")
            raise ValueError("; ".join(details))

        return template.format_map(variables)

    @staticmethod
    def _placeholders(template: str) -> set[str]:
        placeholders: set[str] = set()
        try:
            for _, field_name, format_spec, conversion in Formatter().parse(template):
                if field_name is None:
                    continue
                if not PLACEHOLDER_PATTERN.fullmatch(field_name) or format_spec or conversion:
                    raise ValueError(f"invalid prompt placeholder: {field_name}")
                placeholders.add(field_name)
        except ValueError as error:
            if str(error).startswith("invalid prompt placeholder:"):
                raise
            raise ValueError("malformed prompt template") from error
        return placeholders
