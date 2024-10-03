#!/bin/bash
for i in {1..5}
do
    echo $i
    python main_arxiv.py --model gnn --interp 1 --rebasin 1 --log_steps 100
done

