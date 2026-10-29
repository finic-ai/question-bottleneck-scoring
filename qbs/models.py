import json
import time
import urllib.error
import urllib.request


class Grayson:
    name = "grayson-1"
    url = "https://api.finic.ai/v1/decide"
    batch = 128

    def __init__(self, api_key):
        self.api_key = api_key

    def answer(self, record, questions):
        probs = []
        for i in range(0, len(questions), self.batch):
            chunk = questions[i:i + self.batch]
            body = {
                "model": self.name,
                "data": record,
                "questions": {f"q{j}": {"type": "noul", "instructions": q} for j, q in enumerate(chunk)},
            }
            answers = self._post(body)["answers"]
            probs += [answers[f"q{j}"]["noul"] for j in range(len(chunk))]
        return probs

    def _post(self, body):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        request = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise RuntimeError(f"Grayson returned {e.code}: {e.read().decode()}") from None
            except urllib.error.URLError:
                if attempt == 4:
                    raise
            time.sleep(2 ** attempt)


class Other:
    name = "other"

    def answer(self, record, questions):
        raise NotImplementedError("Make Other.answer in qbs/models.py return the probability of yes for each question.")


def load(config):
    if config["model"] == "grayson":
        return Grayson(config["api_key"])
    return Other()
