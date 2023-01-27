<!-- REPRODUCE CALIBERATOR -->
nohup python calib_exps/run_exp.py > log/calib_log 2>&1 &

<!-- 1. FEW-SHOT -->
nohup python few_shot.py --show_result --model gpt3 --num_dev 1000 > log/few_shot_log 2>&1 &

<!-- 1. EXPLAIN AND PREDICT -->
nohup python manual_joint.py --show_result --model gpt3 --num_dev 1000 > log/cot_log 2>&1 &

<!-- 1. EXPLAIN AND PREDICT + CONSISTENCY -->
nohup python consistency.py --show_result --model gpt3 --num_dev 1000 --plot_consistency > log/consistency_log 2>&1 &

<!-- 2. GENERATE VERIFYING QUESTIONS -->
nohup python verifying_questions.py --show_result --model gpt3 --run_length_test > log/verifying_questions_log 2>&1 &

<!-- 3. RETRIEVE CONTEXT -->
nohup python relevant_context.py --show_result --model gpt3 --num_dev 1000 --retrieval wikipedia > log/relevant_context_wikipedia_2_log 2>&1 &

To use other options, use --retrieval from options ['wikipedia', 'drqa', 'google', 'dataset']
To use the google option, make sure you run google.py first.

<!-- 4. GENERATE VERIFYING ANSWERS -->
nohup python verifying_answers.py --show_result --model gpt3 --num_dev 1000 --retrieval wikipedia > log/verifying_answers_wikipedia_log 2>&1 &

<!-- 5. ANSWERING AGAIN -->
nohup python answer_again.py --show_result --model gpt3 --retrieval wikipedia --num_dev 1000 > log/answering_again_wikipedia_log 2>&1 &