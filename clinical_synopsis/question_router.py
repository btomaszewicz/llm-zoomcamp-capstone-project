from dataclasses import dataclass


@dataclass
class RouteResult:
    question_type: str | None
    confidence: float
    scores: dict[str, float]


def route_question(question: str) -> RouteResult:
    q = question.lower()

    rules = {
        "patient_overview": [
            ("overview", 2.5),
            ("summary", 2.0),
            ("background", 1.5),
            ("history", 1.0),
            ("status", 0.5),
            ("current status", 1.5),
        ],
        "conditions": [
            ("condition", 2.5),
            ("conditions", 2.5),
            ("diagnosis", 2.0),
            ("diagnosed", 2.0),
            ("problem list", 1.5),
            ("comorbid", 1.5),
        ],
        "medications": [
            ("medication", 2.5),
            ("medications", 2.5),
            ("drug", 1.5),
            ("therapy", 1.5),
            ("prescription", 1.5),
            ("taking", 1.0),
            ("current meds", 2.0),
        ],
        "oncology_timeline": [
            ("oncology", 3.0),
            ("cancer", 2.5),
            ("tumor", 2.0),
            ("chemo", 2.5),
            ("chemotherapy", 3.0),
            ("radiation", 2.0),
            ("stage", 1.5),
            ("progression", 2.0),
            ("response", 1.5),
            ("treatment history", 2.0),
        ],
    }

    scores: dict[str, float] = {}

    for question_type, patterns in rules.items():
        score = 0.0
        for phrase, weight in patterns:
            if phrase in q:
                score += weight
        scores[question_type] = score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_type, best_score = ranked[0]
    second_score = ranked[1][1]

    # confidence heuristic: strong absolute score and clear margin
    margin = best_score - second_score
    confidence = min(1.0, best_score / 5.0)

    # make oncology stricter due to false positives with "history", "stage", or "progression"
    thresholds = {
        "patient_overview": 0.5,
        "conditions": 0.5,
        "medications": 0.5,
        "oncology_timeline": 0.55,
    }

    if best_score < 2.0 or margin < 1.0 or confidence < thresholds[best_type]:
        return RouteResult(question_type=None, confidence=confidence, scores=scores)

    return RouteResult(question_type=best_type, confidence=confidence, scores=scores)


# # for separate routing logic, we can use this function to determine if the question needs clarification
# def needs_clarification(route: RouteResult) -> bool:
#     if route.question_type is None:
#         return True
#     threshold = ONCOLOGY_CONFIDENCE_THRESHOLD if route.question_type == "oncology_timeline" else DEFAULT_CONFIDENCE_THRESHOLD
#     return route.confidence < threshold
