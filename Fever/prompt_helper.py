import numpy as np
_PROMPT_HEADER = 'Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFO.'
QUESTION_PROMPT = 'Write a question that validates the reason for a claim.\n\nClaim:  Reg Watson is a current television producer.\nReason: Reginald James Watson AM was an Australian television producer and screenwriter.\nQuestion: What is Reg Watson\'s occupation?\n\nClaim: The Gadsden flag was named by Christopher Gadsden.\nReason: there is no information on who named the Gadsden flag.\nQuestion: Who named the Gadsden flag?\n\n'
QUESTION_PROMPT_no_claim = 'Write a question that validates the claim.\n\nClaim:  Reginald James Watson AM was an Australian television producer and screenwriter.\nQuestion: What is Reg Watson\'s occupation?\n\nClaim: there is no information on who named the Gadsden flag.\nQuestion: Who named the Gadsden flag?\n\n'
VA_PROMPT = "Reginald James Watson AM (27 August 1926 – 8 October 2019) was an Australian television producer and screenwriter. He was executive producer on Crossroads and created Australian media exports serials such as Prisoner, Neighbours, The Young Doctors and Sons and Daughters.\nQuestion: What is Reg Watson's occupation?\nAnswer: Reg Watson was an Australian television producer and screenwriter\n\nThe flag is named after politician Christopher Gadsden (1724–1805), who designed it in 1775 during the American Revolution.\nQuestion: Who named the Gadsden flag?\nAnswer: The Gadsden flag is named after Christopher Gadsden, but there is no information on who named it.\n\n"
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
    x = x.lstrip()
    if x.lower() == 'supports': x = 'SUPPORTS'
    if x.lower() == 'refutes': x = 'REFUTES' 
    if x.lower() == 'not enough info': x = 'NOT ENOUGH INFO'
    if x.lower() == 'not enough information': x = 'NOT ENOUGH INFO'
    return x

class JointPrompHelper:
    style = None
    def __init__(self, style):
        self.label_leading_token = None
        self.style = style

    def prompt_for_joint_prediction(self, ex, shots):
        raise NotImplementedError()

    def post_process(self, p):
        self.post_process_prediction(p)
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
        completion_tokens = tokens[completion_start_tok_idx:(completion_end_tok_idx)]
        completion_probs = pred["logprobs"]["token_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]
        completed_top_logprobs = pred["logprobs"]["top_logprobs"][completion_start_tok_idx:(completion_end_tok_idx)]

        ans_logprob, rat_logprob, ans_prob_perc = self.extract_answer_and_rationale_logprobs(pred, token_offset, \
            completion_start_tok_idx, completion_tokens, completion_probs, completed_top_logprobs)

        top_choices = pred["logprobs"]["top_logprobs"][completion_start_tok_idx]
        cls_probs = []
        for t in self.label_leading_token:
            if t in top_choices:
                cls_probs.append(np.exp(top_choices[t]))
            else:
                cls_probs.append(.0) 
        pred['class_probs'] = cls_probs

        pred["answer_logprob"] = ans_logprob
        pred["rationale_logprob"] = rat_logprob
        pred["joint_logprob"] = ans_logprob + rat_logprob
        pred["ans_prob_percentage"] = ans_prob_perc
        return ans_logprob, rat_logprob

    def extract_answer_and_rationale(self, p):
        raise NotImplementedError()

    def post_process_prediction(self, p, no_alter = False):
        ans, rat = self.extract_answer_and_rationale(p)
        if not no_alter: #alter
            p["answer"] = ans
            p["rationale"] = rat
            p["label"] = normalize_prediction(ans)
        return ans, rat

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
            "Claim: {}\nA: {}{} {}.\n".format(
                s["question"], s["manual_rationale"], EP_STYLE_SEP,
                s["label"]) for s in shots
        ]
        input_example = "Claim: {}\nA:".format(ex["question"])
        prompt = "\n".join(showcase_examples + [input_example])
        prompt = _PROMPT_HEADER + '\n\n' + prompt
        return prompt, stop_signal

    def extract_answer_and_rationale_logprobs(self, pred, token_offset, completion_start_tok_idx, \
        completion_tokens, completion_probs, top_logprobs):
        sep = get_sep_text(pred)
        if sep == None:            
            return -100, -100, 0
        sep_token_offset = pred["completion_offset"] + pred["text"].index(sep)
        sep_start_idx = token_offset.index(sep_token_offset) - completion_start_tok_idx

        rat_logprob = sum(completion_probs[:sep_start_idx + 3])
        ans_logprob = sum(completion_probs[(sep_start_idx + 3):])
        ans_logprob_li = np.exp(completion_probs[(sep_start_idx + 3):])
        otherans_logprob_li = []
        for oal in top_logprobs[(sep_start_idx + 3):]:
            probs = list(oal.values())
            exped_probs = np.exp(probs)
            otherans_logprob_li.append(np.sum(exped_probs))
        ans_prob_perc = np.mean([a/b for (a, b) in zip(ans_logprob_li, otherans_logprob_li)])
        return ans_logprob, rat_logprob, ans_prob_perc

    def post_process_confidence(self, pred):
        if 'logprobs' not in pred:
            pred["answer_logprob"] = -100
            pred["rationale_logprob"] = -100
            pred["joint_logprob"] = -100
            return -100, -100
        return super().post_process_confidence(pred)

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

    def prompt_for_question_generation(self, question, sentence, no_claim = False):
        stop_signal = "\n\n"
        if no_claim:
            end_question = f"Claim: {sentence}\nQuestion:"
            prompt = f"{QUESTION_PROMPT_no_claim}{end_question}"
        else:
            end_question = f"Claim: {question}\nReason: {sentence}\nQuestion:"
            prompt = f"{QUESTION_PROMPT}{end_question}"
        return prompt, stop_signal

    def prompt_for_verifying_answers(self, pars, verifying_question):
        stop_signal = "\n\n"
        par = ' '.join(pars).replace('\n', '')
        end_question = "{}\nQuestion: {}\nAnswer:".format(par, verifying_question)
        prompt = f"{VA_PROMPT}{end_question}"
        return prompt, stop_signal

    def prompt_for_answering_again(self, ex, shots, rationale):
        stop_signal = "\n\n"
        showcase_examples = [
            "Claim: {}\nA: {}{} {}.\n".format(
                s["question"], s["manual_rationale"], EP_STYLE_SEP,
                s["label"]) for s in shots
        ]
        input_example = "Claim: {}\nA: {}".format(ex["question"], rationale)
        prompt = "\n".join(showcase_examples + [input_example])
        prompt = _PROMPT_HEADER + '\n\n' + prompt
        return prompt, stop_signal