CHECKPOINT_DIR="/home/work/im2txt/model/train"
VOCAB_FILE="/home/work/im2txt/model/word_counts.txt"
./bazel-bin/im2txt/inference_server --checkpoint_path=${CHECKPOINT_DIR} --vocab_file=${VOCAB_FILE}
