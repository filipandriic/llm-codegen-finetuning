python napravi_dataset.py \
  --roots "/Users/filipandric/Downloads/Moji projekti_pia_ip_psz/ip" "/Users/filipandric/Downloads/Moji projekti_pia_ip_psz/pia" "/Users/filipandric/Downloads/Moji projekti_pia_ip_psz/psz" \
  --out dataset_srb.jsonl \
  --modes file diff \
  --samples-per-project 40 \
  --max-relevant-files 4 \
  --max-file-bytes 6000