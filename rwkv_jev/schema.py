from dataclasses import dataclass
import json
import math


def serialize(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class Question:
    kind: str
    instructions: object
    names: tuple[str, ...]
    descriptions: tuple[object, ...]

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Each question must be an object")
        kind = value.get("type")
        instructions = value.get("instructions")
        if not isinstance(instructions, (str, dict, list)) or not instructions:
            raise ValueError("instructions must be a nonempty string, object or array")
        if isinstance(instructions, str) and not instructions.strip():
            raise ValueError("instructions must not be blank")
        criteria = value.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError("choice requires 1–255 candidates")
            if any(not isinstance(name, str) or not name.strip() for name in criteria):
                raise ValueError("Candidate names must be nonempty strings")
            names, descriptions = tuple(criteria), tuple(criteria.values())
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 255:
                raise ValueError("score requires 2–255 ordered levels")
            names = tuple(str(index) for index in range(len(criteria)))
            descriptions = tuple(criteria)
        elif kind == "noul":
            criteria = {} if criteria is None else criteria
            if not isinstance(criteria, dict) or set(criteria) - {"false", "true"}:
                raise ValueError("noul criteria may contain only false and true")
            names = ("false", "true")
            descriptions = (criteria.get("false", "No"), criteria.get("true", "Yes"))
        else:
            raise ValueError(f"Unknown question type: {kind!r}")
        for description in descriptions:
            if description is not None and not isinstance(description, (str, dict, list)):
                raise ValueError("Descriptions must be strings, objects, arrays or null")
        serialize([instructions, descriptions])
        return cls(kind, instructions, names, descriptions)

    def prefix(self):
        return "\nQuestion: " + serialize({"type": self.kind, "instructions": self.instructions})

    def candidates(self):
        return ["\nCandidate: " + serialize({"name": name, "description": description}) + "\nSuitability:"
                for name, description in zip(self.names, self.descriptions)]

    def target(self, value):
        if self.kind == "noul":
            if not isinstance(value, bool):
                raise ValueError("noul label must be a boolean")
            return int(value)
        if self.kind == "score":
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(self.names):
                raise ValueError("score label must be a valid integer level")
            return value
        if value not in self.names:
            raise ValueError("choice label must name a candidate")
        return self.names.index(value)

    def answer(self, probabilities):
        if len(probabilities) != len(self.names) or any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise ValueError("Invalid candidate probabilities")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise ValueError("Candidate probabilities must sum to one")
        if self.kind == "noul":
            return {"type": "noul", "noul": probabilities[1]}
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        count = len(probabilities)
        confidence = 1.0 if count == 1 else (count * probabilities[winner] - 1) / (count - 1)
        result = {"type": self.kind, "probabilities": dict(zip(self.names, probabilities)),
                  "confidence": max(0.0, min(1.0, confidence))}
        if self.kind == "choice":
            result["choice"] = self.names[winner]
        else:
            result["score"] = sum(index * value for index, value in enumerate(probabilities))
            result["legend"] = dict(zip(self.names, self.descriptions))
        return result
