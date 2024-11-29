#!/bin/bash

SESSION_NAME="train_hyperyolox"

tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION_NAME
    tmux send-keys -t $SESSION_NAME "source ~/.bashrc" C-m
    tmux send-keys -t $SESSION_NAME "conda create -n hyperyl python=3.10 -y" C-m
    tmux send-keys -t $SESSION_NAME "conda activate hyperyl" C-m
    tmux send-keys -t $SESSION_NAME "pip3 install -r requirements2.txt" C-m
    tmux send-keys -t $SESSION_NAME "cd Credit" C-m
    tmux send-keys -t $SESSION_NAME "cd Hyper-YOLO" C-m
    tmux send-keys -t $SESSION_NAME "python3 ultralytics/models/yolo/detect/train.py" C-m
else
    echo "Session $SESSION_NAME already exists."
fi

tmux attach -t $SESSION_NAME
