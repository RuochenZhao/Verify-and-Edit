import random
random.seed(10)

QUESTION_PROMPT = "Write a question that validates the reason for an overall question.\n\nOverall Question: What is the date of death of the composer of film Baalaraajana Kathe?\nReason: First, the composer of film Baalaraajana Kathe is S. Hanumantha Rao.\nQuestion: Who is the composer of film Baalaraajana Kathe?\n\nOverall Question: Who lived longer, Edward Frederick Sanderson or Forrest Towns?\nReason: First, Edward Frederick Sanderson died at age 81.\nQuestion: How long did Edward Frederick Sanderson live for?\n\n"
VA_PROMPT = 'The film was released in 1984 by Essex Films. Kistimaat is a 2014 Bangladeshi action film directed by Ashiqur Rahman and produced by Tiger Media Limited and The Abhi Pictures. I\'m Taraneh, 15 is a 2002 Iranian film directed by Rasul Sadrameli. The film was released on May 4, 2001.\nQuestion: When was the film Kistimaat released?\nAnswer: The film Kistimaat was released in 2014.\n\nDwaram Venkataswami Naidu and also a lyricist. The film has musical score by S. Hanumantha Rao. Rao died 27 May 1980. Rao married Raja Mani with whom he had three daughters and one son.\nQuestion: Who is the composer of film Baalaraajana Kathe?\nAnswer: The composer of film Baalaraajana Kathe is S. Hanumantha Rao.\n\nAdib Kheir was a leading Syrian nationalist of the 1920s. Filmed on location in the Purcell Mountains in British Columbia, the film was directed by Frank Marshall, written by John Patrick Shanley, and narrated by John Malkovich. Frank Wilton Marshall( born September 13, 1946) is an American film producer and director, often working in collaboration with his wife, Kathleen Kennedy. He received the Irving G. Thalberg award from the Academy of Motion Picture Arts and Sciences in 2018.\nQuestion: Who is the director of film Alive (1993 Film)?\nAnswer: The director of film Alive is Frank Marshall.\n\n'
# PROMOT CONTROL
EP_STYLE_SEP = " The answer is"
EP_POSSIBLE_SEP_LIST = [
    " The answer is",
    " First, the answer is",
    " Second, the answer is",
    " Third, the answer is"
]

def get_joint_prompt_helper(style):
    if style == "e-p":
        return JointEandPPromptHelper(style)
    else:
        raise RuntimeError("Unsupported prompt style")

def get_sep_text(pred):
    for sep in EP_POSSIBLE_SEP_LIST:
        if sep in pred["text"]:
            return sep
    return None

def normalize_prediction(x):
    x = x.strip()
    if x.lower() == 'false': x = 'False'
    if x.lower() == 'incorrect': x = 'False'
    if x.lower() == 'not true': x = 'False'
    if x.lower() == 'true': x = 'True' 
    if x.lower() == 'correct': x = 'True' 
    return x

class JointPrompHelper:
    style = None
    def __init__(self, style):
        self.label_leading_token = None
        self.style = style

    def prompt_for_joint_prediction(self, ex, shots):
        raise NotImplementedError()

    def post_process(self, p, change_rationale = True):
        self.post_process_prediction(p, change_rationale = change_rationale)
        self.post_process_confidence(p)

    def post_process_confidence(self, pred):
        completion_offset = pred["completion_offset"]
        tokens = pred["logprobs"]["tokens"]
        token_offset = pred["logprobs"]["text_offset"]

        completion_start_tok_idx = token_offset.index(completion_offset)
        # exclusive idxs
        if "<|endoftext|>" in tokens:
            completion_end_tok_idx = tokens.index("<|endoftext|>") + 1
        else:
            completion_end_tok_idx = len(tokens)
        completion_probs = pred["logprobs"]["token_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]

        sep_text = get_sep_text(pred)
        if sep_text is not None:
            sep_token_offset = completion_offset + pred["text"].index(sep_text)
            sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx

            rat_logprob = sum(completion_probs[:sep_start_idx + 3])
            ans_logprob = sum(completion_probs[(sep_start_idx + 3):-1])
        else:
            ans_logprob = sum(completion_probs)
            rat_logprob = 0

        pred["answer_logprob"] = ans_logprob
        pred["rationale_logprob"] = rat_logprob
        pred["joint_lobprob"] = ans_logprob + rat_logprob
        return ans_logprob, rat_logprob

    def extract_answer_and_rationale(self, p):
        text = p["text"].strip()
        # print(text)
        sep = get_sep_text(p)
        if sep is not None:
            segments = text.split(sep)
            if len(segments) == 1:
                answer = segments[0].replace(sep.strip(), '').strip().strip('.')
                rationale = 'none'
            else:
                answer = segments[1].strip().strip('.')
                rationale = segments[0].strip()
        else:
            answer = 'NOT ENOUGH INFO'
            rationale = 'none'
        return answer, rationale

    def post_process_prediction(self, p, change_rationale = True):
        text = p["text"]
        text = text.strip()

        # place holder
        answer = "null"
        rationale = "null"
        rationale_indices = []
        answer, rationale = self.extract_answer_and_rationale(p)
    
        p["answer"] = answer
        if change_rationale:
            p["rationale"] = rationale
            p["rationale_indices"] = rationale_indices
        return answer, rationale

    def extract_answer_and_rationale_logprobs(self):
        raise NotImplementedError()

class JointEandPPromptHelper(JointPrompHelper):
    def __init__(self, style):
        super().__init__(style)
        self.sep = EP_STYLE_SEP
        self.label_leading_token = [' SUP', ' RE', ' NOT']

    def prompt_for_joint_prediction(self, ex, shots):
        stop_signal = "\n\n"
        showcase_examples = [
            "Question: {}\nA: {}{} {}.\n".format(
                s["question"], s["manual_rationale"], EP_STYLE_SEP,
                s["answer"]) for s in shots
        ]
        input_example = "Question: {}\nA:".format(ex["question"])
        prompt = "\n".join(showcase_examples + [input_example])
        return prompt, stop_signal

    def extract_answer_and_rationale_logprobs(self, pred, token_offset, completion_start_tok_idx, completion_tokens, completion_probs):
        sep = get_sep_text(pred)
        if sep == None:            
            return -100, -100
        sep_token_offset = pred["completion_offset"] + pred["text"].index(sep)
        sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx

        rat_logprob = sum(completion_probs[:sep_start_idx + 3])
        ans_logprob = sum(completion_probs[(sep_start_idx + 3):])
        return ans_logprob, rat_logprob

    def prompt_for_question_generation(self, question, sentence):
        stop_signal = "\n\n"
        end_question = f"Overall Question: {question}\nReason: {sentence}\nQuestion:"
        prompt = f"{QUESTION_PROMPT}{end_question}"
        return prompt, stop_signal

    def prompt_for_verifying_answers(self, shots, pars, verifying_question):
        stop_signal = "\n\n"
        par = ' '.join(pars).replace('\n', '')
        input_example = "{}\nQuestion: {}\Answer:".format(par, verifying_question)
        prompt = VA_PROMPT + input_example
        return prompt, stop_signal

    def prompt_for_answering_again(self, ex, shots, rationale):
        stop_signal = "\n\n"
        showcase_examples = [
            "Question: {}\nA: {}{} {}.\n".format(
                s["question"], s["manual_rationale"], EP_STYLE_SEP,
                s["answer"]) for s in shots
        ]
        input_example = "Question: {}\nA: {}".format(ex["question"], rationale)
        prompt = "\n".join(showcase_examples + [input_example])
        return prompt, stop_signal