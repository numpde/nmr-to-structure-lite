onmt_translate \
  --model ../results/001_big/run/model_step_110000.pt \
  --src ./001_big_src-test_n30.txt \
  --output ./001_big_pre-test_n30.txt \
  --n_best 10 \
  --beam_size 10 \
  --gpu -1 \
  --verbose \
  | grep -Ev "FutureWarning|def (forward|backward)"

sed 's/ //g' 001_big_pre-test_n30.txt > 001_big_pre-test_n30.txt.no-space.txt
sed 's/ //g' 001_big_tgt-test_n30.txt > 001_big_tgt-test_n30.txt.no-space.txt

# Note, beam_size=777 requires about 85GB of RAM
