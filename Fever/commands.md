<!-- BASELINE: DIRECTLY PREDICT THE ANSWER -->
CUDA_VISIBLE_DEVICES=6 nohup python few_shot.py --train_slice 0  --num_dev 3000 --show_result --run_pred --use_sampled > log/sampled_davinci3_predict_log 2>&1 &

<!-- EXPLAIN AND PREDICT -->
CUDA_VISIBLE_DEVICES=6 nohup python manual_joint.py --train_slice 0 --num_dev 3000 --show_result --run_pred --use_sampled > log/sampled_davinci3_joint_log 2>&1 &

<!-- 1. EXPLAIN AND PREDICT + CONSISTENCY -->
CUDA_VISIBLE_DEVICES=4 nohup python consistency.py --train_slice 0 --show_result --model gpt3 --num_dev 3000 --use_sampled > log/sampled_davinci3_consistency_log 2>&1 &

<!-- 2. GENERATE VERIFYING QUESTIONS -->
CUDA_VISIBLE_DEVICES=1 nohup python verifying_questions.py --train_slice 0 --show_result --num_dev 3000 --use_sampled > log/sampled_davinci3_verifying_questions_log 2>&1 &

<!-- 3. FIND RELEVANT CONTEXT -->
CUDA_VISIBLE_DEVICES=3 nohup python relevant_context.py --train_slice 0 --show_result --retrieval wikipedia  --num_dev 3000 --plot_numbers --check_inclusion --use_sampled > log/sampled_davinci3_relevant_context_wikipedia_log 2>&1 &

<!-- 4. GENERATE VERIFYING ANSWERS -->
CUDA_VISIBLE_DEVICES=3 nohup python verifying_answers.py --train_slice 0 --show_result --num_dev 3000 --retrieval wikipedia --use_sampled > log/sampled_davinci3_verifying_answers_wikipedia_log 2>&1 &

<!-- 5. ANSWER AGAIN -->
CUDA_VISIBLE_DEVICES=1 nohup python answer_again.py --train_slice 0 --show_result --model gpt3 --mode answer_with_context --retrieval wikipedia --num_dev 3000 --use_sampled > log/sampled_davinci3_knowledge_wikipedia_log 2>&1 &

<!-- REPRODUCE CALIBERATOR -->
CUDA_VISIBLE_DEVICES=2 nohup python run_calib_exp.py --with_context > log/davinci3_calib_log 2>&1 &
