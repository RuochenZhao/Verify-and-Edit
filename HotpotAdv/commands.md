<!-- BASELINE: DIRECTLY PREDICT THE ANSWER -->
nohup python few_shot.py --train_slice 0 --show_result --run_pred > log/gpt3_predict_log 2>&1 &

<!-- EXPLAIN AND PREDICT -->
nohup python manual_joint.py --train_slice 0 --show_result --run_pred > log/gpt3_joint_log 2>&1 &

<!-- 1. EXPLAIN AND PREDICT + CONSISTENCY -->
nohup python consistency.py --train_slice 0 --show_result --model gpt3 --plot_consistency > log/gpt3_consistency_log 2>&1 &

<!-- 2. GENERATE VERIFYING QUESTIONS -->
nohup python verifying_questions.py --train_slice 0 --show_result --consistency_threshold 3 > log/davinci3_verifying_questions_log 2>&1 &

<!-- 3. FIND RELEVANT CONTEXT -->
nohup python relevant_context.py --show_result --wikipedia > log/relevant_context_wikipedia_log 2>&1 &

To use other systems, try options: --DPR --drqa --google, or leave no options for using the dataset
To use google, make sure you run google.py first.

<!-- 4. GENERATE VERIFYING ANSWERS -->
nohup python verifying_answers.py --train_slice 0 --show_result --wikipedia > log/verifying_answers_log 2>&1 &

<!-- 5. ANSWER AGAIN -->
nohup python answer_again.py --train_slice 0 --show_result --model gpt3 --mode answer_with_context --consistency_threshold 3 --wikipedia > log/gpt3_knowledge_wikipedia_log 2>&1 &

<!-- REPRODUCE CALIBERATOR -->
CUDA_VISIBLE_DEVICES=2 nohup python run_calib_exp.py > log/gpt3_calib_log 2>&1 &