#!/bin/sh
rm -f .env
touch .env
echo "UID=$(id -u $USER)" >> .env
echo "GID=$(id -g $USER)" >> .env
echo "UNAME=$USER" >> .env