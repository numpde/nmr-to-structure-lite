ssh ubuntu@150.136.93.242 \
'find /home/ubuntu/fs/tmp/nmr-to-structure-lite/ -type f -mmin +5 -printf "%P\n"' | \
rsync -av --progress --ignore-existing \
--files-from=- ubuntu@150.136.93.242:/home/ubuntu/fs/tmp/nmr-to-structure-lite/ \
/home/ra/repos/nmr-to-structure-lite/results/
