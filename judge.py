"""
judge.py

Automatic relevance judge, run on every conversation turn -- this is a
direct port of 05-monitoring/code/judge.py. In production you don't have
a ground-truth "original answer" to compare against (see the course's
13-llm-as-judge.md: "In production, we usually don't have that original
answer for real user questions"), so this judge only looks at the
question and the generated answer, same as the course.

Runs on Groq via evaluation_utils.get_client() -- Groq's Responses API
supports client.responses.parse(text_format=PydanticModel) directly
(confirmed in Groq's docs for openai/gpt-oss-120b, the model this
project defaults to), so llm_structured_retry() needed no changes to
work here, just a different client and model.

app.py calls evaluate_relevance() right after agent_loop() and stores
the verdict in the feedback table with source='judge', alongside
source='user' thumbs up/down -- see db_feedback.py.
"""

from typing import Literal

from pydantic import BaseModel

from evaluation_utils import get_client, llm_structured_retry


class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str


judge_instructions = """
You are an expert evaluator for a stock market assistant that answers
questions using live data from tools (price snapshots, historical bars,
company info, news, dividends, market status).

Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer directly addresses the question with concrete,
  specific data (prices, dates, percentages, named articles, etc.)
- PARTLY_RELEVANT: the answer is on-topic but vague, incomplete, or
  missing some of what was asked
- NON_RELEVANT: the answer does not address the question, or invents
  data instead of using real figures
""".strip()

judge_prompt = """
Question: {question}
Generated Answer: {answer}
""".strip()


def evaluate_relevance(question: str, answer: str, client=None):
    if client is None:
        client = get_client()

    prompt = judge_prompt.format(question=question, answer=answer)

    result, usage = llm_structured_retry(
        client,
        judge_instructions,
        prompt,
        RelevanceVerdict,
    )

    return result.relevance, result.explanation


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    q = "What's Apple's current stock price?"
    a = "AAPL is currently trading at $231.42, up 1.2% from yesterday's close of $228.66."

    relevance, explanation = evaluate_relevance(q, a)
    print(relevance)
    print(explanation)
