#!/bin/bash
# Deploy kuroko-voice-webapp to XServer VPS

SERVER="root@162.43.4.11"
KEY="~/.ssh/xserver_vps_rsa"
REMOTE_DIR="/root/kuroko-voice-webapp"

echo "Deploying to XServer VPS..."

# Deploy Python files
scp -i $KEY main.py rag.py requirements.txt $SERVER:$REMOTE_DIR/

# Deploy static files
scp -i $KEY static/* $SERVER:$REMOTE_DIR/static/

# Restart service
ssh -i $KEY $SERVER "systemctl restart kuroko-interview"

echo "Done! Service restarted."
