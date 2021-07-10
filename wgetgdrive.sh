#!/bin/bash
# Get file (>100MB) from Google Drive
# ./wgetgdrive.sh $1 $2
ID=$1
FILE_NAME=$2
URL="https://docs.google.com/uc?export=download&id=$ID"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$ID" -O $FILE_NAME && rm -rf /tmp/cookies.txt