import argparse
import re
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import datasets

from openrlhf.utils.logging_utils import init_logger
from transformers import AutoTokenizer
from symeval import EvaluatorMathBatch

logger = init_logger(__name__)


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def extract_answer_math(s):
    """
    input query
    """
    pattern = r"Answer:(.*)<\|eot_id\|>"
    match = re.search(pattern, s)
    if match:
        ans = match.group(1)
    else:
        ans = "failedmatch"
    return ans


def normalize_text(text):
    """
    Normalize text by lowercasing and removing special characters
    """
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()


class MathRuleProxy:
    def __init__(self, args):
        # eval_dataset in the format of 
        # {"question": q, "messages": [], "answer": a}
        if args.data_path.endswith(".jsonl"):
            with open(args.data_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            eval_dataset = [json.loads(line) for line in lines]
        else:
            eval_dataset = datasets.load_from_disk(args.data_path)['train'].to_list()
        self.eval_data_dict = self.get_answer_dict(eval_dataset)
        logger.info(f"len(train_data_dict): {len(self.eval_data_dict)}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.log_file = args.log_file
        self.avg_length_dict = []
        self.cnt = 0
        self.avg_len = 5000
        self.key_words = [
            "wait",
            "double check",
            "what",
            "how",
            "why",
            "alternatively",
            "think",
            "rethink",
            "?",
            "change",
            "try",
            "check",
        ]

    def get_answer_dict(self, eval_dataset):
        """Get answer dict from eval dataset
        Args:
            eval_dataset (list): List of dictionaries containing question and answer pairs
        Returns:
            dict: Dictionary with normalized questions as keys and answers as values
            eval_data_dict = {question: answer}
        """
        eval_data_dict = {}
        for item in eval_dataset:
            # eval_data_dict[normalize_text(item["question"])] = item["answer"]
            # since the answer for musique comes in the form of a list, all the elements of the list are correct 
            if "answers" in item:
                assert type(item["answers"]) == list, f"Key answers not in list format for {item['question']}"
                assert len(item["answers"]) > 0, f"Empty answers for {item['question']}"
                eval_data_dict[normalize_text(item["question"])] = [normalize_text(ans) for ans in item["answers"]]
            else:
                eval_data_dict[normalize_text(item["question"])] = normalize_text(item["answer"])
        return eval_data_dict

    def get_qa(self, query):
        # question = query.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0].strip()
        # question = question.replace(
        #     "Please reason step by step, and put your final answer within \\boxed{}", ""
        # ).strip()
        # solution = query.split("<｜Assistant｜>")[-1].strip()
        import re

        text = query
        pattern = r"Question: (.*) <\|start_header_id\|>"
        match = re.search(pattern, text)
        if match:
            input_part = match.group(1)
            # logger.info(f"input_part: {input_part}")
            question = input_part
        else:
            question = "What is 1+1"
        # question = normalize_text(question)
        solution_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*)<\|eot_id\|>"
        match = re.search(solution_pattern, text)
        if match:
            solution = match.group(1)
        else:
            solution = "No answer"
        # solution = normalize_text(solution)
        logger.info("Question is: ")
        logger.info(question)
        logger.info("Solution is: ")
        logger.info(solution)
        logger.info("End of QA")

        return question, solution

    def get_query_answer(self, query):
        # query = query.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0].strip()
        # query = query.replace("Please reason step by step, and put your final answer within \\boxed{}", "").strip()
        # query = 
        import re

        text = query
        pattern = r"Question:(.*)<\|start_header_id\|>"
        match = re.search(pattern, text)
        if match:
            input_part = match.group(1)
            logger.info(f"input_part: {input_part}")
            query = input_part
        else:
            query = "No answer"
        query = normalize_text(query)
        # print(query)
        return self.eval_data_dict.get(query, "No answer")

    def get_query_pred(self, query):
        return extract_answer_math(query)

    def get_thought(self, solution):
        thought = solution.split("<think>")[-1].strip().split("</think>")[0].strip()
        return thought

    def get_reward(self, queries):
        preds = []
        answers = []
        questions = []
        solutions = []
        finished_lst = []
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
            question, solution = self.get_qa(queries[i])
            preds.append(self.get_query_pred(solution))
            answers.append(self.get_query_answer(question))
            questions.append(question)
            solutions.append(solution)
        print(preds, answers)

        # evaluator = EvaluatorMathBatch()
        # scores = evaluator.batch_eq(ref_answers=answers, pred_answers=preds)
        scores = []
        logger.info(f"answers: {answers}")
        logger.info(f"preds: {preds}")
        for single_answer, single_pred in zip(answers, preds):
            if type(single_answer) == list:
                for ans in single_answer:
                    if ans in single_pred:
                        scores.append(1.0)
                        break
                scores.append(0.0)
            elif type(single_answer) == str:
                if single_answer in single_pred:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            else:
                raise ValueError(f"Type of single_answer is {type(single_answer)}")
        length_scores = []
        pattern_scores = []
        for i, query in enumerate(queries):
            self.cnt = self.cnt + 1
            if "Answer:" not in solutions[i]:
                scores[i] = -2.0
                finished_lst.append("0")
            else:
                if not scores[i]:
                    scores[i] = -1.0
                    finished_lst.append("1")
                else:
                    scores[i] = 1.0
                    finished_lst.append("1")

            if "Answer:" not in query:
                length_scores.append(1)
            else:
                length_scores.append(0)

        # Write query-score pairs to JSONL if log_file is provided
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for q, a, s, f_f in zip(
                    questions,
                    solutions,
                    scores,
                    finished_lst,
                ):
                    record = {
                        "question": q,
                        "solution": a,
                        "score": s,
                        "finished": f_f,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # return scores
        assert len(scores) == len(length_scores)
        # final_score = [[s0, s1] for s0, s1 in zip(scores, length_scores)]
        # TODO: check how does the final_score work given 2 scores in a list
        final_score = [s0 for s0 in scores]
        return final_score
        # return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--port", type=int, default=5001, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--log_file", type=str, default=None, help="Path to JSONL log file")

    args = parser.parse_args()

    # server
    reward_model = MathRuleProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        # convert data to JSON
        # logger.info(f"Received JSON: {data}")
        # print("Received JSON: ")
        # print(data)
        with open("/mnt/longcontext/models/siyuan/test_code/LongRL/data.json", "w") as f:
            json.dump(data, f)
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
