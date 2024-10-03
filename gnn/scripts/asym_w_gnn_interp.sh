#!/bin/bash
for i in {1..5}
do
    echo $i
    python main_arxiv.py --model asym_w_gnn --interp 1 --log_steps 100
done

